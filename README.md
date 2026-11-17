# Champion–Challenger Sentiment Model Comparison with Hugging Face and MLflow

**Student:** Nandini Trivedi · **USN:** 1MS25CI093 · **Course:** Skill Lab (MLOps with MLflow)

## 1. Task and models

| Item | Value |
|---|---|
| Task | Binary sentiment classification (POSITIVE / NEGATIVE) |
| Dataset | SST-2 (`stanfordnlp/sst2`), validation split |
| Subset | 200 sentences, randomly sampled with seed 42 (saved in `data/eval_inputs.csv`) |
| Model A, current Champion | `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters) |
| Model B, Challenger | `textattack/bert-base-uncased-SST-2` (~110M parameters, outputs `LABEL_0` / `LABEL_1`, mapped to NEGATIVE / POSITIVE) |
| Registered model name | `sentiment-model` |
| MLflow experiment | `hf-sentiment-champion-challenger` |

## 2. How to run

```bash
# 0. One-time setup
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 1. Terminal 1: start the MLflow tracking server (leave it running)
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --workers 1

# 2. Terminal 2: run the experiment, register the models, predict on new data
python evaluate.py
python register_champion.py
python predict_from_mlflow.py

# Optional (stretch goal): swap the aliases and rerun the same script
python register_champion.py --flip
python predict_from_mlflow.py
```

The MLflow UI is at http://127.0.0.1:5000.

### Files

| File | Purpose |
|---|---|
| `config.py` | All settings: USN, models, sample size, seed and the Champion rule |
| `sentiment_model.py` | Custom `mlflow.pyfunc.PythonModel` wrapping the Hugging Face pipeline (as in Lab 9) |
| `evaluate.py` | Runs both models on the same inputs and logs runs, metrics, artifacts and models to MLflow |
| `register_champion.py` | Registers both models and sets the `@champion` / `@challenger` aliases |
| `predict_from_mlflow.py` | Loads `models:/sentiment-model@champion` and predicts on `data/new_data.csv` |

## 3. Champion rule (written before the final run)

1. **Primary metric:** macro-F1 on the 200-sentence evaluation set. **Higher is better.**
2. **Minimum improvement margin:** the Challenger replaces the Champion only if its macro-F1 is **at least 0.01 (1 percentage point) higher**. Otherwise the Champion stays. Reason: with 200 sentences, one sentence is 0.5 percentage points, so a gap smaller than about 2 sentences could easily be noise.
3. **Tie-breaker:** if the difference is smaller than the margin in either direction, the **faster model** (lower `avg_latency_ms`) is chosen.
4. **Guardrail:** the Challenger is rejected if its average latency is **more than 5× the Champion's**, whatever its score.
5. **Direction of every metric:**

| Metric | Direction |
|---|---|
| accuracy, precision, recall, f1, macro_f1 | higher is better |
| avg_latency_ms, total_inference_s | lower is better |

## 4. Results

<!-- Paste the table printed at the end of evaluate.py here -->

| Metric | A (Champion) | B (Challenger) | B − A | Better |
|---|---|---|---|---|
| accuracy | | | | higher |
| precision | | | | higher |
| recall | | | | higher |
| f1 | | | | higher |
| macro_f1 | | | | higher |
| total_inference_s | | | | lower |
| avg_latency_ms | | | | lower |

**Plots** (also logged in MLflow): `outputs/comparison_bar.png`, `outputs/confusion_matrix_distilbert-sst2.png`, `outputs/confusion_matrix_bert-base-sst2.png`

### Disagreements (5 cases where A and B disagree)

<!-- Pick 5 rows from outputs/disagreements.csv and write a short comment of your own for each -->

| Sentence | Reference | A says | B says | My comment |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

## 5. Decision

<!-- 150–250 words, written by you AFTER seeing results. Answer:
     - Who is the Champion and why (quote the rule)?
     - Does the Challenger improve on the Champion? By how much (delta macro-F1)?
     - Is the difference meaningful with only 200 samples?
     - Trade-offs: speed (latency ratio), size (67M vs 110M), types of errors (see confusion matrices) -->

## 6. Limitations and declaration

**Limitations**
- Only 200 sentences, so differences of about 1–2 points may be noise.
- SST-2 contains short movie-review sentences, so results may not carry over to other kinds of text.
- Both models were fine-tuned on SST-2 training data, so they are evaluated in-domain.
- Latency was measured on one laptop CPU and depends on hardware and background load.
- The logged MLflow model stores the Hugging Face model ID and loads the weights from the Hugging Face cache or Hub when it is loaded.

**Declaration**
- Tools used: Python, MLflow, Hugging Face Transformers, PyTorch, scikit-learn, pandas, matplotlib.
- AI assistance: I used Claude (Anthropic) to help structure the project and write the initial code. I ran every script myself, checked the outputs, wrote the decision and the disagreement comments, and can explain every line.
