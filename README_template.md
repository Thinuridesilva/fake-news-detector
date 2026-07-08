# Fake News Detector

A text classifier that labels news articles as **real** or **fake**, comparing a classical
TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT model.

🔗 **Live demo:** _\<paste your Hugging Face Space link here\>_

## Dataset

[Kaggle "Fake and Real News Dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
— ~45,000 articles split across `Fake.csv` and `True.csv`.

## Approach

1. **EDA** — checked class balance, text length distributions, top bigrams per class, and
   checked for data leakage (the `subject` column maps almost 1:1 to label in the raw data,
   so it was dropped from training features).
2. **Preprocessing** — lowercased, stripped URLs/numbers/punctuation, removed stopwords.
   Stratified 80/20 train/test split.
3. **Baseline** — TF-IDF (10k features, unigrams+bigrams) + Logistic Regression.
4. **DistilBERT** — fine-tuned `distilbert-base-uncased` for 3 epochs on a free Kaggle GPU.

## Results

| Model            | Accuracy | F1   | AUC  | Train time |
|-------------------|----------|------|------|------------|
| TF-IDF + LogReg   | _fill in_ | _fill in_ | _fill in_ | < 1 min |
| DistilBERT        | _fill in_ | _fill in_ | _fill in_ | ~20-25 min (GPU) |

_Fill in real numbers from `eval_outputs/model_comparison.csv` after running both models._

## Key findings

- _Write 1-2 sentences on the class balance / text length finding from Day 3-4._
- _Write 2 paragraphs from the error analysis (Day 13-14): what kinds of articles get
  misclassified, and why. Common patterns to look for: satire mistaken for fake news,
  short neutral-toned factual pieces mistaken for fake, or near-duplicate articles split
  across train/test._
- _Note honestly if DistilBERT only modestly beat the baseline — that's a legitimate and
  interesting finding given how stylistically distinct the two source CSVs are._

## Project structure

```
.
├── data/                  # Fake.csv, True.csv (not committed), cleaned/train/test CSVs
├── src/
│   ├── data_loader.py     # Day 2
│   ├── eda.py              # Day 3-5
│   ├── preprocessing.py    # Day 6-7
│   ├── train_baseline.py   # Day 8
│   ├── tokenize_for_bert.py # Day 9-10 (run on Kaggle/Colab)
│   ├── train_distilbert.py  # Day 11-12 (run on Kaggle/Colab)
│   └── evaluate.py         # Day 13-14
├── app.py                  # Day 15-16 — Gradio demo
├── requirements.txt
└── README.md
```

## Running it yourself

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# put Fake.csv and True.csv in data/, then:
python src/preprocessing.py
python src/train_baseline.py
python src/evaluate.py
python app.py
```

For the DistilBERT model, upload `train.csv`/`test.csv` to a Kaggle dataset, enable a GPU
accelerator, and run `tokenize_for_bert.py` then `train_distilbert.py` there.

## What I'd improve next

- _e.g. test on a different fake-news dataset to see if it generalizes, since this one's
  two classes come from visibly different sources._
