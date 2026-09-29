import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


DATA = Path("data/processed/data.npz")
MODEL = Path("models/model.h5")
METRICS = Path("metrics.json")
CONFUSION_MATRIX = Path("reports/confusion_matrix.png")


def main() -> None:
    data = np.load(DATA)
    model = tf.keras.models.load_model(MODEL)
    loss, accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    predictions = np.argmax(model.predict(data["x_test"], verbose=0), axis=1)
    matrix = confusion_matrix(data["y_test"], predictions)

    METRICS.write_text(
        json.dumps({"test_loss": float(loss), "test_accuracy": float(accuracy)}, indent=2) + "\n",
        encoding="utf-8",
    )
    CONFUSION_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Fashion-MNIST Confusion Matrix")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX, dpi=150)
    plt.close(figure)
    print(json.dumps({"test_loss": float(loss), "test_accuracy": float(accuracy)}, indent=2))


if __name__ == "__main__":
    main()
