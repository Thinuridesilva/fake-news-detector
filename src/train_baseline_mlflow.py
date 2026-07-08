"""
Day 8 (MLflow version) -- TF-IDF + Logistic Regression baseline with experiment tracking.
Run: python src/train_baseline_mlflow.py
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_experiment("fake-news-detector")


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df["clean_text"] = train_df["clean_text"].fillna("")
    test_df["clean_text"] = test_df["clean_text"].fillna("")

    with mlflow.start_run(run_name="tfidf-logreg"):
        # params
        max_features = 10000
        ngram_range = (1, 2)
        max_iter = 1000

        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("model_type", "TF-IDF + LogisticRegression")

        # train
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = vectorizer.fit_transform(train_df["clean_text"])
        X_test = vectorizer.transform(test_df["clean_text"])

        clf = LogisticRegression(max_iter=max_iter)
        clf.fit(X_train, train_df["label"])

        # evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(test_df["label"], preds)
        f1 = f1_score(test_df["label"], preds)
        auc = roc_auc_score(test_df["label"], probs)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)

        print(classification_report(test_df["label"], preds, target_names=["real", "fake"]))
        print(f"AUC: {auc:.4f}")

        # save model
        joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
        joblib.dump(clf, MODEL_DIR / "logreg_model.joblib")
        mlflow.sklearn.log_model(clf, "logreg_model")

        print(f"MLflow run logged. Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")


if __name__ == "__main__":
    main()
