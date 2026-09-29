from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


INPUT = Path("data/raw/data.npz")
OUTPUT = Path("data/processed/data.npz")


def main() -> None:
    with Path("params.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)["preprocess"]

    raw = np.load(INPUT)
    x_train = raw["x_train"].astype("float32") / 255.0
    x_test = raw["x_test"].astype("float32") / 255.0
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        raw["y_train"],
        test_size=config["validation_size"],
        random_state=config["seed"],
        stratify=raw["y_train"],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=raw["y_test"],
    )
    print(f"Saved processed arrays to {OUTPUT}")


if __name__ == "__main__":
    main()
