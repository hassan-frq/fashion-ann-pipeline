"""
Part B2 — src/preprocess.py
Loads the raw arrays from data/raw/, normalizes pixel values to [0, 1],
and splits a validation set out of the training data. Saves the
resulting train/val/test arrays under data/processed/.

Hyperparameters (test_size, seed) come from params.yaml -> preprocess.

Run standalone:
    python src/preprocess.py
"""

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()["preprocess"]
    test_size = params["test_size"]
    seed = params["seed"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_raw = np.load(os.path.join(RAW_DIR, "train.npz"))
    test_raw = np.load(os.path.join(RAW_DIR, "test.npz"))

    x_train_full = train_raw["images"].astype("float32") / 255.0
    y_train_full = train_raw["labels"]
    x_test = test_raw["images"].astype("float32") / 255.0
    y_test = test_raw["labels"]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=test_size,
        random_state=seed,
        stratify=y_train_full,
    )

    np.savez(os.path.join(PROCESSED_DIR, "train.npz"), images=x_train, labels=y_train)
    np.savez(os.path.join(PROCESSED_DIR, "val.npz"), images=x_val, labels=y_val)
    np.savez(os.path.join(PROCESSED_DIR, "test.npz"), images=x_test, labels=y_test)

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    print(f"Processed data written to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
