"""
Part B4 — src/evaluate.py
Loads the trained model and the processed test set. Computes test loss
and accuracy, generates a confusion matrix image, and writes all
metrics to metrics.json at the project root.

Run standalone:
    python src/evaluate.py
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
METRICS_PATH = "metrics.json"
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    x_test, y_test = test_data["images"], test_data["labels"]

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "model.h5"))

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Fashion-MNIST ANN")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Metrics written to {METRICS_PATH}")
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
