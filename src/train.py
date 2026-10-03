"""
Part B3 — src/train.py
Builds a Sequential ANN: Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax).
Compiles with Adam and sparse_categorical_crossentropy. Trains on
data/processed/, then saves the trained model to models/model.h5 and the
training history to models/history.csv.

Hyperparameters (dense_units, dropout_rate, learning_rate, epochs, batch_size)
come from params.yaml -> train.

Run standalone:
    python src/train.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(dense_units, dropout_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    params = load_params()["train"]
    dense_units = params["dense_units"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    os.makedirs(MODELS_DIR, exist_ok=True)

    train_data = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
    val_data = np.load(os.path.join(PROCESSED_DIR, "val.npz"))

    x_train, y_train = train_data["images"], train_data["labels"]
    x_val, y_val = val_data["images"], val_data["labels"]

    model = build_model(dense_units, dropout_rate, learning_rate)
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    model.save(os.path.join(MODELS_DIR, "model.h5"))
    pd.DataFrame(history.history).to_csv(os.path.join(MODELS_DIR, "history.csv"), index=False)

    print(f"Model saved to {MODELS_DIR}/model.h5")
    print(f"History saved to {MODELS_DIR}/history.csv")


if __name__ == "__main__":
    main()
