"""
predict_from_mlflow.py - loads the Champion from the Model Registry BY ALIAS
and predicts on brand-new sentences (data/new_data.csv).

Important: this script never imports transformers or calls pipeline().
Everything goes through MLflow. The script does not know (or care) which
version is the Champion - it asks the registry.

Usage:
    python predict_from_mlflow.py                      # uses @champion
    python predict_from_mlflow.py --alias challenger   # optional
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="champion",
                        choices=["champion", "challenger"])
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()
    name = config.REGISTERED_MODEL_NAME
    model_uri = f"models:/{name}@{args.alias}"

    # 1. Ask the registry which version currently holds this alias
    mv = client.get_model_version_by_alias(name, args.alias)
    hf_id_from_registry = mv.tags.get("hf_model_id", "unknown")

    # 2. Load the model through MLflow (alias URI, no run ID)
    print(f"Loading {model_uri} ...")
    model = mlflow.pyfunc.load_model(model_uri)
    hf_id_from_model = model.model_config.get("hf_model_id", "unknown")

    print("\n========== MODEL LOADED ==========")
    print(f"Model URI              : {model_uri}")
    print(f"Registered model       : {name}")
    print(f"Version                : {mv.version}")
    print(f"HF model ID (registry) : {hf_id_from_registry}")
    print(f"HF model ID (metadata) : {hf_id_from_model}")
    print(f"Source run ID          : {mv.run_id}")

    # 3. Predict on the new sentences
    new_df = pd.read_csv("data/new_data.csv")
    preds = model.predict(new_df[["text"]])

    out = new_df.copy()
    out["predicted_label"] = preds["predicted_label"].values
    out["score"] = preds["score"].values
    if "expected_label" in out.columns:
        out["correct"] = out["predicted_label"] == out["expected_label"]
    out["alias"] = args.alias
    out["model_version"] = mv.version
    out["hf_model_id"] = hf_id_from_model

    # 4. Print a clean table and save it
    show_cols = [c for c in ["text", "expected_label", "predicted_label",
                             "score", "correct"] if c in out.columns]
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print("\n========== PREDICTIONS ON NEW DATA ==========")
        print(out[show_cols].to_string(index=False))

    if "correct" in out.columns:
        print(f"\nCorrect: {out['correct'].sum()} / {len(out)}")

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / "new_data_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
