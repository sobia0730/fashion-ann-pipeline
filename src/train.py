from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


INPUT = Path("data/processed/data.npz")
MODEL_OUTPUT = Path("models/model.h5")
HISTORY_OUTPUT = Path("models/history.csv")


def main() -> None:
    with Path("params.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)["train"]

    tf.keras.utils.set_random_seed(42)
    data = np.load(INPUT)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(config["dense_units"], activation="relu"),
            tf.keras.layers.Dropout(config["dropout_rate"]),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=2,
    )

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUTPUT)
    import csv

    with HISTORY_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(history.history.keys())
        writer.writerows(zip(*history.history.values()))
    print(f"Saved model to {MODEL_OUTPUT} and history to {HISTORY_OUTPUT}")


if __name__ == "__main__":
    main()
