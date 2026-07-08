"""
Day 8 -- TF-IDF + Logistic Regression baseline.
This is the number every later model needs to beat.

Run: python src/train_baseline.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df["clean_text"] = train_df["clean_text"].fillna("")
    test_df["clean_text"] = test_df["clean_text"].fillna("")

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_df["label"])

    preds = clf.predict(X_test)
    print(classification_report(test_df["label"], preds, target_names=["real", "fake"]))

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, MODEL_DIR / "logreg_model.joblib")
    print(f"Saved baseline model + vectorizer to {MODEL_DIR}")


if __name__ == "__main__":
    main()
