"""
data_utils.py

Shared helper for loading the train/validation/test CSVs into
features (X) and labels (y), used by every training/evaluation script
so this logic isn't duplicated across files.
"""

from pathlib import Path

import pandas as pd


def get_feature_columns() -> list[str]:
    return [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def get_data_dir() -> Path:
    # training/ -> ml -> ai -> app -> backend
    return Path(__file__).resolve().parent.parent.parent.parent.parent / "data"


def load_split(csv_name: str):
    """
    Loads one of train.csv / validation.csv / test.csv and returns
    (X, y) - features and labels, ready for scikit-learn.
    """
    path = get_data_dir() / csv_name
    df = pd.read_csv(path)
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["label"]
    return X, y