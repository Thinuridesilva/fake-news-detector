"""
Day 2 -- Load the Kaggle "Fake and Real News Dataset" and take a first look.

BEFORE running this:
1. Go to https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download Fake.csv and True.csv
3. Put both files in the `data/` folder at the root of this repo

Run: python src/data_loader.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    fake = pd.read_csv(data_dir / "Fake.csv")
    real = pd.read_csv(data_dir / "True.csv")

    fake["label"] = 1  # 1 = fake
    real["label"] = 0  # 0 = real

    df = pd.concat([fake, real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.info())
    print(df.head())
    print("\nLabel counts:\n", df["label"].value_counts())
