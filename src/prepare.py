from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist


OUTPUT = Path("data/raw/data.npz")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    np.savez_compressed(
        OUTPUT,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"Saved raw dataset to {OUTPUT}")


if __name__ == "__main__":
    main()
