"""
Day 3-5 -- EDA: class balance, text length, vocabulary, duplicates, leakage check.

Run: python src/eda.py
Saves plots and printouts into eda_outputs/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from data_loader import load_data

OUT_DIR = Path(__file__).resolve().parent.parent / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)


def class_balance(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    print("Class balance (0=real, 1=fake):\n", counts)
    counts.plot(kind="bar", title="Class distribution (0=real, 1=fake)")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.savefig(OUT_DIR / "class_balance.png")
    plt.close()


def length_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().apply(len)

    for col in ["text_length", "word_count"]:
        plt.figure()
        for label, name in [(0, "real"), (1, "fake")]:
            df[df.label == label][col].plot(kind="hist", bins=50, alpha=0.5, label=name)
        plt.legend()
        plt.title(f"{col} distribution by class")
        plt.savefig(OUT_DIR / f"{col}_hist.png")
        plt.close()

    print(df.groupby("label")[["text_length", "word_count"]].mean())
    return df


def vocab_analysis(df: pd.DataFrame) -> None:
    for label, name in [(0, "real"), (1, "fake")]:
        text = " ".join(df[df.label == label]["text"].astype(str).tolist())
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        wc.to_file(OUT_DIR / f"wordcloud_{name}.png")

    vec = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=20)
    X = vec.fit_transform(df["text"].astype(str))
    sums = X.sum(axis=0)
    bigrams = sorted(zip(vec.get_feature_names_out(), sums.tolist()[0]), key=lambda x: -x[1])
    print("Top bigrams overall:\n", bigrams)


def duplicate_and_leakage_check(df: pd.DataFrame) -> None:
    dupes = df.duplicated(subset=["text"]).sum()
    print(f"Duplicate articles (exact text match): {dupes}")

    # IMPORTANT: in this specific dataset, the `subject` column almost
    # perfectly predicts the label (e.g. all "politicsNews" is real,
    # all "Government News" is fake). That's a leakage shortcut a model
    # could exploit instead of learning real signal -- worth dropping
    # subject/date columns before training.
    if "subject" in df.columns:
        print("\nSubject vs label crosstab (1:1 mapping here = leakage risk):")
        print(pd.crosstab(df["subject"], df["label"]))

    if "date" in df.columns:
        print("\nDate range per class:")
        print(df.groupby("label")["date"].agg(["min", "max"]))


if __name__ == "__main__":
    df = load_data()
    class_balance(df)
    df = length_analysis(df)
    vocab_analysis(df)
    duplicate_and_leakage_check(df)
    print(f"\nAll plots saved to {OUT_DIR}")
