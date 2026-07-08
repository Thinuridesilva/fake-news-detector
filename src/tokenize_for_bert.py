import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

DATA_DIR = "/content/drive/MyDrive/fake-news-detector/data"


def load_and_tokenize(
    train_path: str = f"{DATA_DIR}/train.csv",
    test_path: str = f"{DATA_DIR}/test.csv",
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = pd.read_csv(train_path)[["clean_text", "label"]].rename(columns={"clean_text": "text"})
    test_df = pd.read_csv(test_path)[["clean_text", "label"]].rename(columns={"clean_text": "text"})

    # Guard against empty strings becoming NaN after CSV round-trip
    train_df["text"] = train_df["text"].fillna("").astype(str)
    test_df["text"] = test_df["text"].fillna("").astype(str)

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.remove_columns(["text"]).rename_column("label", "labels")
    test_ds = test_ds.remove_columns(["text"]).rename_column("label", "labels")

    # NOTE: do NOT call set_format("torch") here -- triggers a torchvision
    # VideoReader import bug in the current Colab environment. The Trainer
    # handles tensor conversion automatically without it.

    return train_ds, test_ds, tokenizer


if __name__ == "__main__":
    train_ds, test_ds, tokenizer = load_and_tokenize()
    print(train_ds)
    print(test_ds)
