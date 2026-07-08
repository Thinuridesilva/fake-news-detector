import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from tokenize_for_bert import MODEL_NAME, load_and_tokenize

DRIVE_DIR = "/content/drive/MyDrive/fake-news-detector"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main() -> None:
    train_ds, test_ds, tokenizer = load_and_tokenize()

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=f"{DRIVE_DIR}/distilbert_fakenews",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(f"{DRIVE_DIR}/distilbert_fakenews/best_model")
    tokenizer.save_pretrained(f"{DRIVE_DIR}/distilbert_fakenews/best_model")
    print(f"Saved best model to {DRIVE_DIR}/distilbert_fakenews/best_model")


if __name__ == "__main__":
    main()
