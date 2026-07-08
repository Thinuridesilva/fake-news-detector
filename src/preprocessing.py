"""
Day 6-7 -- Clean text, save cleaned dataset, train/test split.

Run: python src/preprocessing.py
"""

import re
import string
from pathlib import Path

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_data

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOPWORDS = set(stopwords.words("english"))
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # strip URLs
    text = re.sub(r"\d+", "", text)  # strip standalone numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # strip punctuation
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)


if __name__ == "__main__":
    df = load_data()
    df["clean_text"] = df["text"].apply(clean_text)

    df.to_csv(DATA_DIR / "cleaned_news.csv", index=False)
    print(f"Saved cleaned dataset: {DATA_DIR / 'cleaned_news.csv'} ({len(df)} rows)")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print("Train label balance:\n", train_df["label"].value_counts(normalize=True))
    print("Test label balance:\n", test_df["label"].value_counts(normalize=True))
