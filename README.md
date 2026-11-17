# Champion–Challenger Sentiment Model Comparison with Hugging Face and MLflow

**Student:** PRACHI CHANDRA · **USN:** 1MS25CI093 · **Course:** Skill Lab (MLOps with MLflow)

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

Evaluation set: 200 SST-2 validation sentences (seed 42), identical for both models.

| Metric | A: DistilBERT (old Champion) | B: BERT-base (Challenger) | B − A | Better |
|---|---|---|---|---|
| accuracy | 0.9150 | 0.9500 | +0.0350 | higher |
| precision | 0.8947 | 0.9450 | +0.0502 | higher |
| recall | 0.9533 | 0.9626 | +0.0093 | higher |
| f1 | 0.9231 | 0.9537 | +0.0306 | higher |
| **macro_f1 (primary)** | **0.9141** | **0.9497** | **+0.0356** | higher |
| total_inference_s | 3.0541 | 5.7817 | +2.7276 | lower |
| avg_latency_ms | 15.27 | 28.91 | +13.64 (1.89× slower) | lower |

Correct predictions: A = 183/200, B = 190/200. The models disagree on 11 of 200 sentences (B correct on 9, A correct on 2).

**Prediction on new data** (`data/new_data.csv`, 8 sentences not from SST-2): the Champion (version 2) got 6/8 correct. Both models fail on the same two sentences: one with implied negativity ("This phone heats up so much that I can barely hold it.") and one sarcastic sentence ("Oh great, another surprise quiz on a Monday morning.").

**Stretch goal:** after `python register_champion.py --flip`, the unchanged `predict_from_mlflow.py` loaded version 1 (DistilBERT) instead of version 2, which shows that the script follows the alias and does not need to know which version is the Champion. The aliases were then flipped back. See `screenshots/predict_output_flipped.png`.

**Plots** (also logged in MLflow): `outputs/comparison_bar.png`, `outputs/confusion_matrix_distilbert-sst2.png`, `outputs/confusion_matrix_bert-base-sst2.png`

### Disagreements (5 cases where A and B disagree)

The models disagree on 11 of 200 sentences (full list in `outputs/disagreements.csv`). B is right on 9 and A on 2. Five representative cases:

| idx | Sentence | Reference | A says | B says | Comment |
|---|---|---|---|---|---|
| 683 | true tale of courage -- and complicity -- at auschwitz is a harrowing drama that tries to tell of the unspeakable . | POSITIVE | NEGATIVE | POSITIVE ✅ | Dark topic words ("harrowing", "unspeakable") mislead A. B sees that the review praises the film. |
| 770 | i do n't think i laughed out loud once . | NEGATIVE | POSITIVE | NEGATIVE ✅ | Negation: A reacts to "laughed" and misses the "don't". B handles the negation. |
| 356 | jason x is positively anti-darwinian : nine sequels and 400 years later , the teens are none the wiser and jason still kills on auto-pilot . | NEGATIVE | POSITIVE | NEGATIVE ✅ | The word "positively" tricks A, but the sentence mocks the film. |
| 143 | a solid film ... but more conscientious than it is truly stirring . | POSITIVE | POSITIVE ✅ | NEGATIVE | Mixed review: praise followed by "but". B focuses on the criticism after "but". This one is genuinely borderline. |
| 422 | it moves quickly , adroitly , and without fuss ; it does n't give you time to reflect on the inanity -- and the cold war datedness -- of its premise . | POSITIVE | POSITIVE ✅ | NEGATIVE | Praise for the pacing plus a dig at the premise. B weighs the negative words ("inanity", "datedness") too heavily. |

**Pattern:** A tends to react to individual words ("positively", "laughed", "harrowing"). B handles negation and context better, but it can over-weight criticism in mixed reviews.

## 5. Decision

**Champion: Model B, `textattack/bert-base-uncased-SST-2` (registry version 2, alias `@champion`).**
**Challenger: Model A, `distilbert-base-uncased-finetuned-sst-2-english` (version 1, alias `@challenger`).**

Following the rule written before the final run, the Challenger replaced the Champion. B's macro-F1 is 0.9497 against A's 0.9141, a gain of +0.0356 (3.56 percentage points), which is well above the 1-point margin. B is 1.89× slower (28.9 ms vs 15.3 ms per sentence). That is inside the 5× latency guardrail, so the guardrail does not block the switch, and the tie-breaker was not needed.

In plain counts, B got 190 of 200 sentences right and A got 183. The two models disagreed on only 11 sentences: B was right on 9 and A on 2. This advantage is consistent, but it rests on a small number of sentences. A McNemar test on these 11 disagreements gives p ≈ 0.07, so the improvement is suggestive rather than statistically conclusive at this sample size. A larger evaluation set would be needed to confirm it.

Most of the gain comes from precision (0.945 vs 0.895): A labels more negative sentences as positive. Recall is almost the same (0.963 vs 0.953). The trade-offs are model size (about 110M vs 67M parameters) and speed (almost twice as slow). Neither model handled sarcasm or implied negativity in the new examples.

## 6. Limitations and declaration

**Limitations**
- Only 200 sentences, so differences of about 1–2 points may be noise.
- SST-2 contains short movie-review sentences, so results may not carry over to other kinds of text.
- Both models were fine-tuned on SST-2 training data, so they are evaluated in-domain.
- Latency was measured on one laptop CPU and depends on hardware and background load.
- The logged MLflow model stores the Hugging Face model ID and loads the weights from the Hugging Face cache or Hub when it is loaded.
- The significance claim (McNemar p ≈ 0.07) is based on only 11 disagreeing sentences.
- The new-data check uses only 8 hand-written sentences; both models failed on sarcasm and implied negativity, which SST-2 rarely tests.

**Declaration**
- Tools used: Python, MLflow, Hugging Face Transformers, PyTorch, scikit-learn, pandas, matplotlib.
- AI assistance: I used Claude (Anthropic) to help structure the project, write the initial code and draft the results and decision text. I ran every script myself, checked the outputs against the MLflow UI, reviewed and edited the written sections, and can explain every line.
