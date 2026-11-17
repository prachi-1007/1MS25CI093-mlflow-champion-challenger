"""
evaluate.py - runs Model A and Model B on the SAME inputs and logs
everything to MLflow.

Creates three runs in the experiment:
    compare_sentiment                    (parent / summary run)
      |- A_sentiment_distilbert-sst2     (Champion candidate)
      |- B_sentiment_bert-base-sst2      (Challenger candidate)

Run it with the MLflow server already running:
    python evaluate.py
"""

import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from mlflow.models import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

import config
from sentiment_model import SentimentPipelineModel, normalize_label

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}


# ---------------------------------------------------------------------------
# Step 1: fix every random seed so the run can be repeated exactly
# ---------------------------------------------------------------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


# ---------------------------------------------------------------------------
# Step 2: build the shared evaluation set and save it
# ---------------------------------------------------------------------------
def load_eval_set():
    ds = load_dataset(config.DATASET_ID, split=config.DATASET_SPLIT)
    ds = ds.shuffle(seed=config.SEED).select(range(config.EVAL_SAMPLES))
    df = pd.DataFrame({
        "idx": ds["idx"],
        "sentence": ds["sentence"],
        "label": ds["label"],
    })
    df["reference"] = df["label"].map(LABEL_NAMES)
    path = DATA_DIR / "eval_inputs.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} evaluation sentences to {path}")
    return df


# ---------------------------------------------------------------------------
# Step 3: run one model on all sentences and time it
# ---------------------------------------------------------------------------
def run_model(hf_model_id, sentences):
    pipe = pipeline(config.HF_TASK, model=hf_model_id, device="cpu")
    pipe(sentences[:1], truncation=config.TRUNCATION)   # warm-up, not timed

    start = time.perf_counter()
    outputs = pipe(sentences, batch_size=config.BATCH_SIZE,
                   truncation=config.TRUNCATION)
    total_s = time.perf_counter() - start

    preds = [normalize_label(o["label"]) for o in outputs]
    scores = [float(o["score"]) for o in outputs]
    return preds, scores, total_s


# ---------------------------------------------------------------------------
# Step 4: metrics (POSITIVE is treated as the positive class)
# ---------------------------------------------------------------------------
def compute_metrics(references, predictions, total_s):
    n = len(references)
    return {
        "accuracy": accuracy_score(references, predictions),
        "precision": precision_score(references, predictions, pos_label="POSITIVE"),
        "recall": recall_score(references, predictions, pos_label="POSITIVE"),
        "f1": f1_score(references, predictions, pos_label="POSITIVE"),
        "macro_f1": f1_score(references, predictions, average="macro"),
        "total_inference_s": total_s,
        "avg_latency_ms": total_s / n * 1000,
    }


def confusion_matrix_figure(references, predictions, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        references, predictions, labels=["NEGATIVE", "POSITIVE"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def comparison_figure(metrics_a, metrics_b, name_a, name_b):
    quality = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                   gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(quality))
    w = 0.38
    ax1.bar(x - w / 2, [metrics_a[m] for m in quality], w, label=f"A: {name_a}")
    ax1.bar(x + w / 2, [metrics_b[m] for m in quality], w, label=f"B: {name_b}")
    ax1.set_xticks(x, quality)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Quality metrics (higher is better)")
    ax1.legend(loc="lower right")

    ax2.bar(["A", "B"], [metrics_a["avg_latency_ms"], metrics_b["avg_latency_ms"]],
            color=["tab:blue", "tab:orange"])
    ax2.set_title("Avg latency ms (lower is better)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 5: apply the Champion rule written in README / config.py
# ---------------------------------------------------------------------------
def decide(metrics_a, metrics_b):
    pm = config.PRIMARY_METRIC
    delta = metrics_b[pm] - metrics_a[pm]                 # Challenger minus Champion
    lat_ratio = metrics_b["avg_latency_ms"] / metrics_a["avg_latency_ms"]

    if lat_ratio > config.LATENCY_GUARDRAIL:
        return "A", delta, (f"Guardrail: Challenger is {lat_ratio:.1f}x slower "
                            f"(limit {config.LATENCY_GUARDRAIL}x), Champion stays.")
    if delta >= config.MARGIN:
        return "B", delta, (f"Challenger beats Champion on {pm} by {delta:+.4f} "
                            f"(>= margin {config.MARGIN}).")
    if delta <= -config.MARGIN:
        return "A", delta, (f"Champion is better on {pm} by {-delta:.4f} "
                            f"(>= margin {config.MARGIN}).")
    faster = "A" if metrics_a["avg_latency_ms"] <= metrics_b["avg_latency_ms"] else "B"
    return faster, delta, (f"Difference {delta:+.4f} is within the margin "
                           f"{config.MARGIN}; tie-breaker picks the faster model ({faster}).")


def library_tags():
    return {
        "student_usn": config.STUDENT_USN,
        "dataset_name": f"{config.DATASET_ID}:{config.DATASET_SPLIT}",
        "library_versions": (f"transformers={transformers.__version__}; "
                             f"torch={torch.__version__}; mlflow={mlflow.__version__}"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "mlflow_version": mlflow.__version__,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seeds(config.SEED)
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    eval_df = load_eval_set()
    sentences = eval_df["sentence"].tolist()
    references = eval_df["reference"].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}   # key -> dict with preds, scores, metrics, run_id

    with mlflow.start_run(run_name=config.COMPARE_RUN_NAME) as parent:
        mlflow.set_tags(library_tags())

        for key in ["A", "B"]:
            m = config.MODELS[key]
            print(f"\n=== Model {key}: {m['hf_model_id']} ===")
            preds, scores, total_s = run_model(m["hf_model_id"], sentences)
            metrics = compute_metrics(references, preds, total_s)

            with mlflow.start_run(run_name=config.run_name(key), nested=True) as run:
                # Parameters
                mlflow.log_params({
                    "hf_model_id": m["hf_model_id"],
                    "task": config.HF_TASK,
                    "role": f"{m['role']} candidate",
                    "batch_size": config.BATCH_SIZE,
                    "truncation": config.TRUNCATION,
                    "eval_samples": len(sentences),
                    "seed": config.SEED,
                    "device": device,
                })
                # Tags
                mlflow.set_tags(library_tags())
                mlflow.set_tag("hf_model_id", m["hf_model_id"])
                # Metrics
                mlflow.log_metrics(metrics)
                mlflow.log_dict(metrics, "metrics_summary.json")
                # Dataset record (shows under "Datasets" in the UI)
                mlflow.log_input(
                    mlflow.data.from_pandas(eval_df, name="sst2-validation-200"),
                    context="evaluation")

                # Predictions file: input, reference, prediction, score, correct
                pred_df = eval_df[["idx", "sentence", "reference"]].copy()
                pred_df["prediction"] = preds
                pred_df["score"] = [round(s, 4) for s in scores]
                pred_df["correct"] = pred_df["prediction"] == pred_df["reference"]
                pred_path = OUT_DIR / f"predictions_{m['short_name']}.csv"
                pred_df.to_csv(pred_path, index=False)
                mlflow.log_artifact(str(pred_path))
                mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))

                # Plot 1: confusion matrix
                fig = confusion_matrix_figure(references, preds,
                                              f"Model {key}: {m['short_name']}")
                fig.savefig(OUT_DIR / f"confusion_matrix_{m['short_name']}.png", dpi=120)
                mlflow.log_figure(fig, f"confusion_matrix_{m['short_name']}.png")
                plt.close(fig)

                # The model itself, wrapped as a custom pyfunc (Lab 9 style)
                example_in = pd.DataFrame({"text": sentences[:2]})
                example_out = pd.DataFrame({
                    "text": sentences[:2],
                    "predicted_label": preds[:2],
                    "score": [round(s, 4) for s in scores[:2]],
                })
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=SentimentPipelineModel(),
                    code_paths=["sentiment_model.py"],
                    model_config={"hf_model_id": m["hf_model_id"],
                                  "batch_size": config.BATCH_SIZE},
                    signature=infer_signature(example_in, example_out),
                    input_example=example_in,
                    pip_requirements=[
                        f"mlflow=={mlflow.__version__}",
                        f"transformers=={transformers.__version__}",
                        f"torch=={torch.__version__}",
                        "pandas",
                    ],
                )
                mlflow.set_tag("model_uri", info.model_uri)

                results[key] = {"preds": preds, "metrics": metrics,
                                "run_id": run.info.run_id, "model_uri": info.model_uri}
                print({k: round(v, 4) for k, v in metrics.items()})

        # ---------------- Summary run: comparison + decision ----------------
        ma, mb = results["A"]["metrics"], results["B"]["metrics"]
        winner, delta, reason = decide(ma, mb)
        winner_cfg = config.MODELS[winner]

        mlflow.log_params({
            "primary_metric": config.PRIMARY_METRIC,
            "margin": config.MARGIN,
            "latency_guardrail_x": config.LATENCY_GUARDRAIL,
            "tie_breaker": "faster avg_latency_ms",
            "model_A": config.MODELS["A"]["hf_model_id"],
            "model_B": config.MODELS["B"]["hf_model_id"],
        })
        mlflow.log_metric("delta_primary_metric", delta)
        mlflow.log_metric("latency_ratio_B_over_A",
                          mb["avg_latency_ms"] / ma["avg_latency_ms"])
        for name in ma:
            mlflow.log_metric(f"A_{name}", ma[name])
            mlflow.log_metric(f"B_{name}", mb[name])
        mlflow.set_tags({
            "winner": winner,
            "winner_hf_model_id": winner_cfg["hf_model_id"],
            "decision_reason": reason,
        })

        # Plot 2: Champion vs Challenger bar chart
        fig = comparison_figure(ma, mb, config.MODELS["A"]["short_name"],
                                config.MODELS["B"]["short_name"])
        fig.savefig(OUT_DIR / "comparison_bar.png", dpi=120)
        mlflow.log_figure(fig, "comparison_bar.png")
        plt.close(fig)

        # Combined metrics file
        summary = {
            "A": {"hf_model_id": config.MODELS["A"]["hf_model_id"], **ma},
            "B": {"hf_model_id": config.MODELS["B"]["hf_model_id"], **mb},
            "decision": {"primary_metric": config.PRIMARY_METRIC,
                         "margin": config.MARGIN,
                         "delta_primary_metric": delta,
                         "winner": winner, "reason": reason},
            "directions": {k: ("higher is better" if v else "lower is better")
                           for k, v in config.HIGHER_IS_BETTER.items()},
        }
        (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(OUT_DIR / "metrics_summary.json"))

        # Disagreements: sentences where A and B gave different answers
        dis = eval_df[["idx", "sentence", "reference"]].copy()
        dis["pred_A"] = results["A"]["preds"]
        dis["pred_B"] = results["B"]["preds"]
        dis = dis[dis["pred_A"] != dis["pred_B"]]
        dis["comment"] = ""                       # you fill these in yourself
        dis.to_csv(OUT_DIR / "disagreements.csv", index=False)
        mlflow.log_artifact(str(OUT_DIR / "disagreements.csv"))
        mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))
        if Path("README.md").exists():
            mlflow.log_artifact("README.md")
        if Path("requirements.txt").exists():
            mlflow.log_artifact("requirements.txt")

    # ---------------- Print a results table for the README -----------------
    print("\n================ RESULTS (paste into README) ================")
    print("| Metric | A (Champion) | B (Challenger) | B - A | Better |")
    print("|---|---|---|---|---|")
    for name in ma:
        better = "higher" if config.HIGHER_IS_BETTER[name] else "lower"
        print(f"| {name} | {ma[name]:.4f} | {mb[name]:.4f} | "
              f"{mb[name] - ma[name]:+.4f} | {better} |")
    print(f"\nWinner: {winner} ({winner_cfg['hf_model_id']})")
    print(f"Reason: {reason}")
    print(f"Models disagree on {len(dis)} of {len(eval_df)} sentences "
          f"-> outputs/disagreements.csv")
    print("\nNext: python register_champion.py")


if __name__ == "__main__":
    main()
