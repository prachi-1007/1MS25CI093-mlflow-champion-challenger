"""
sentiment_model.py - the custom MLflow model (same idea as Lab 9).

It wraps a Hugging Face sentiment pipeline inside mlflow.pyfunc.PythonModel,
so anyone can load it with mlflow.pyfunc.load_model(...) and call .predict()
without writing any Hugging Face code themselves.

Hugging Face is imported only INSIDE load_context(), i.e. only when MLflow
loads the model. That is allowed by the assignment ("the pyfunc wrapper may
call Hugging Face internally").
"""

import mlflow
import pandas as pd


def normalize_label(raw_label):
    """
    The two models name their outputs differently:
      Model A (distilbert) -> 'POSITIVE' / 'NEGATIVE'
      Model B (textattack) -> 'LABEL_1'  / 'LABEL_0'
    SST-2 uses 1 = positive and 0 = negative, so map both to one format.
    """
    label = str(raw_label).upper()
    if label in ("POSITIVE", "LABEL_1", "1"):
        return "POSITIVE"
    if label in ("NEGATIVE", "LABEL_0", "0"):
        return "NEGATIVE"
    raise ValueError(f"Unexpected label from model: {raw_label}")


class SentimentPipelineModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper around a Hugging Face text-classification pipeline."""

    def load_context(self, context):
        # Runs once when the model is loaded. Settings come from model_config,
        # which was saved together with the model when it was logged.
        from transformers import pipeline   # HF import only happens here

        cfg = context.model_config
        self.hf_model_id = cfg["hf_model_id"]
        self.batch_size = int(cfg.get("batch_size", 16))
        self.pipe = pipeline(
            "text-classification",
            model=self.hf_model_id,
            device="cpu",
        )

    def predict(self, context, model_input, params=None):
        # Accept a DataFrame with a 'text' column (or a plain list of strings)
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]

        outputs = self.pipe(texts, batch_size=self.batch_size, truncation=True)

        return pd.DataFrame({
            "text": texts,
            "predicted_label": [normalize_label(o["label"]) for o in outputs],
            "score": [round(float(o["score"]), 4) for o in outputs],
        })
