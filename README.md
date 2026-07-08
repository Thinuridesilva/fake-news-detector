# Fake News Detector

A text classifier that labels news articles as **REAL** or **FAKE**, comparing a classical
TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT model.
Also includes a production-ready FastAPI service, Docker container, MLflow experiment
tracking, and GitHub Actions CI.

## Results

| Model | Accuracy | F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 99.0% | 99.0% |
| DistilBERT (fine-tuned) | 99.99% | 99.99% |

Both models scored extremely high because the two source CSVs have visibly distinct
writing styles — this is a known characteristic of this dataset, not a sign the models
generalise perfectly to all fake news in the wild.

## Dataset

[Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
~45,000 articles split across `Fake.csv` and `True.csv`.

## Key Findings

**Data leakage discovered:** The `subject` column maps perfectly 1:1 to the label
(e.g. all `politicsNews` articles are real, all `Government News` articles are fake).
A naive model could exploit this shortcut to reach near-100% accuracy without reading
a single word. `subject` was excluded from all training features.

**6,252 duplicate articles** were found in the raw dataset — noted as a data quality
limitation.

**Fake articles average 423 words vs 386 for real** — contrary to the common assumption
that fake news is shorter.

**Error analysis (10 misclassified articles):** Two clear patterns emerged:
- Fake articles written in neutral, journalistic style (citing real organisations,
  quoting officials by name) fooled the baseline model — it learned stylistic patterns,
  not factual truth.
- Short Reuters wire-format real articles (dense, data-heavy, structured) were
  occasionally flagged as fake because they look stylistically unusual compared to
  the average real article in the training set.

**Baseline vs DistilBERT gap is small (0.99%):** On this dataset, the simpler model
was already near-ceiling. This is an honest and interesting finding — it shows the
value of always building a strong baseline before reaching for expensive models.

## Project Structure

├── src/
│   ├── data_loader.py            # Load and merge Fake.csv + True.csv
│   ├── eda.py                    # Class balance, length, vocab, leakage check
│   ├── preprocessing.py          # Clean text, train/test split
│   ├── train_baseline.py         # TF-IDF + Logistic Regression
│   ├── train_baseline_mlflow.py  # Same, with MLflow experiment tracking
│   ├── tokenize_for_bert.py      # Tokenize for DistilBERT
│   ├── train_distilbert.py       # Fine-tune DistilBERT (run on GPU)
│   └── evaluate.py               # Confusion matrix, ROC, error analysis
├── api.py                        # FastAPI service (baseline + DistilBERT endpoints)
├── app.py                        # Gradio demo
├── Dockerfile                    # Container for the FastAPI service
├── requirements.txt              # Training dependencies
├── requirements_api.txt          # API serving dependencies
├── .github/workflows/ci.yml      # GitHub Actions CI — runs on every push
├── eda_outputs/                  # EDA plots
├── eval_outputs/                 # Confusion matrix, misclassified examples
└── models/                       # Trained baseline model + vectorizer

## API Endpoints

Start the server:
```bash
uvicorn api:app --reload --port 8000
```

Or with Docker:
```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

**GET /health**
```json
{"status": "ok", "baseline_loaded": true, "distilbert_loaded": true}
```

**POST /predict**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Federal Reserve announced it would hold interest rates steady."}'
```
```json
{
  "text_preview": "The Federal Reserve announced...",
  "baseline": {"label": "REAL", "confidence": 0.82},
  "distilbert": {"label": "REAL", "confidence": 0.99},
  "agreement": true
}
```

## Running Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# put Fake.csv and True.csv in data/, then:
python src/preprocessing.py
python src/train_baseline.py
python src/evaluate.py
python app.py          # Gradio demo at localhost:7860
uvicorn api:app --reload  # FastAPI at localhost:8000
```

## Tech Stack

- **scikit-learn** — TF-IDF vectorizer, Logistic Regression
- **Hugging Face Transformers** — DistilBERT fine-tuning
- **PyTorch** — deep learning backend
- **FastAPI + uvicorn** — production API serving
- **Docker** — containerisation
- **MLflow** — experiment tracking
- **Gradio** — interactive demo
- **GitHub Actions** — CI on every push

## What I'd Improve Next

- Test on a held-out dataset from a different source to measure real-world generalisation
- Drop duplicate articles before the train/test split to prevent any data leakage
- Add SHAP explanations to the API response to show which words drove the prediction
- Deploy the FastAPI container to Azure and point the Gradio demo at that endpoint