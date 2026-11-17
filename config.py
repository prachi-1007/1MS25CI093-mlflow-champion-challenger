"""
config.py - every setting for the project lives here, in one place.

All three scripts import this file, so a change here (for example the USN or
the sample size) applies everywhere. This file does NOT import Hugging Face.
"""

# ---- Who and where -------------------------------------------------------
STUDENT_USN = "1MS25CI093"
TRACKING_URI = "http://127.0.0.1:5000"          # the MLflow server you started
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"        # name in the Model Registry

# ---- Task and data -------------------------------------------------------
TASK = "sentiment"                               # used in run names
HF_TASK = "text-classification"                  # Hugging Face pipeline task
DATASET_ID = "stanfordnlp/sst2"
DATASET_SPLIT = "validation"                     # the split that has labels
EVAL_SAMPLES = 200                               # subset size
SEED = 42                                        # fixed seed = repeatable sample

# ---- Inference settings (same for both models, so the test is fair) ------
BATCH_SIZE = 16
TRUNCATION = True

# ---- The two candidates --------------------------------------------------
# Model A is the current Champion, Model B is the Challenger.
MODELS = {
    "A": {
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "short_name": "distilbert-sst2",
        "role": "champion",
    },
    "B": {
        "hf_model_id": "textattack/bert-base-uncased-SST-2",
        "short_name": "bert-base-sst2",
        "role": "challenger",
    },
}

# ---- Champion rule (written BEFORE the final run - do not change later) ---
PRIMARY_METRIC = "macro_f1"      # higher is better
MARGIN = 0.01                    # Challenger must win by >= 1 percentage point
LATENCY_GUARDRAIL = 5.0          # reject Challenger if > 5x slower than Champion

# Direction of every metric (True = higher is better)
HIGHER_IS_BETTER = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "macro_f1": True,
    "avg_latency_ms": False,
    "total_inference_s": False,
}


def run_name(key):
    """Run name in the required format, e.g. A_sentiment_distilbert-sst2."""
    return f"{key}_{TASK}_{MODELS[key]['short_name']}"


COMPARE_RUN_NAME = f"compare_{TASK}"
