"""
Part B1 — src/prepare.py
Loads Fashion-MNIST via tf.keras.datasets and saves the raw train/test
images and labels as .npz files under data/raw/.

No hyperparameters needed at this stage, so this script takes no CLI args
and reads nothing from params.yaml.

Run standalone:
    python src/prepare.py
"""

import os
import numpy as np
import tensorflow as tf

RAW_DIR = os.path.join("data", "raw")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    np.savez(os.path.join(RAW_DIR, "train.npz"), images=x_train, labels=y_train)
    np.savez(os.path.join(RAW_DIR, "test.npz"), images=x_test, labels=y_test)

    print(f"Saved raw train set: {x_train.shape}, {y_train.shape}")
    print(f"Saved raw test set:  {x_test.shape}, {y_test.shape}")
    print(f"Raw data written to {RAW_DIR}/")


if __name__ == "__main__":
    main()
