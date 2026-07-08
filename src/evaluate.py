"""
Day 13-14 -- Evaluation + error analysis.

Run: python src/evaluate.py
Assumes you've already run train_baseline.py (so models/ has the
TF-IDF vectorizer + LogisticRegression model). Fill in the DistilBERT
row of the comparison table manually using the numbers Trainer.evaluate()
printed on Kaggle.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent.parent / "eval_outputs"
OUT_DIR.mkdir(exist_ok=True)


def evaluate_baseline(test_df: pd.DataFrame):
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    clf = joblib.load(MODEL_DIR / "logreg_model.joblib")

    X_test = vectorizer.transform(test_df["clean_text"])
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    print("=== TF-IDF + Logistic Regression ===")
    print(classification_report(test_df["label"], preds, target_names=["real", "fake"]))

    cm = confusion_matrix(test_df["label"], preds)
    ConfusionMatrixDisplay(cm, display_labels=["real", "fake"]).plot()
    plt.title("Baseline confusion matrix")
    plt.savefig(OUT_DIR / "baseline_confusion_matrix.png")
    plt.close()

    return preds, probs


def plot_roc(y_true: pd.Series, probs_dict: dict) -> None:
    plt.figure()
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC curves")
    plt.savefig(OUT_DIR / "roc_curves.png")
    plt.close()


def error_analysis(test_df: pd.DataFrame, preds, n: int = 10) -> None:
    wrong = test_df[test_df["label"].values != preds]
    sample = wrong.sample(n=min(n, len(wrong)), random_state=42)
    sample[["text", "label"]].to_csv(OUT_DIR / "misclassified_examples.csv", index=False)
    print(
        f"\nSaved {len(sample)} misclassified examples to "
        "eval_outputs/misclassified_examples.csv -- open this file, read each "
        "row, and write 2 paragraphs in your README on the patterns you find "
        "(e.g. satire flagged as fake, short neutral articles flagged as fake)."
    )


def comparison_table(results: dict) -> None:
    df = pd.DataFrame(results).T
    df.to_csv(OUT_DIR / "model_comparison.csv")
    print("\nComparison table:\n", df)


if __name__ == "__main__":
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df["clean_text"] = test_df["clean_text"].fillna("")
    preds, probs = evaluate_baseline(test_df)
    error_analysis(test_df, preds)

    # Plug in the real DistilBERT numbers once you've run it on Kaggle.
    results = {
        "TF-IDF + LogReg": {"accuracy": None, "f1": None, "auc": None, "train_time_min": "<1"},
        "DistilBERT": {"accuracy": None, "f1": None, "auc": None, "train_time_min": "~20-25"},
    }
    comparison_table(results)
