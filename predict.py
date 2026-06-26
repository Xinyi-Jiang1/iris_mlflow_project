from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd

from src.model import RAW_FEATURE_NAMES


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "_runtime"
MLFLOW_RUNTIME = RUNTIME / "mlflow"
OUTPUTS = RUNTIME / "outputs"
DEFAULT_TRACKING_URI = f"sqlite:///{(MLFLOW_RUNTIME / 'mlflow.db').as_posix()}"
DEFAULT_SAMPLE = [5.1, 3.5, 1.4, 0.2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the latest MLflow decision-tree model and predict one sample.")
    parser.add_argument(
        "--sample",
        nargs=4,
        type=float,
        default=DEFAULT_SAMPLE,
        metavar=("SEPAL_LEN", "SEPAL_WIDTH", "PETAL_LEN", "PETAL_WIDTH"),
    )
    parser.add_argument("--model-uri", default=None)
    return parser.parse_args()


def read_latest_model_uri() -> str:
    latest_model_uri = OUTPUTS / "latest_model_uri.txt"
    if not latest_model_uri.exists():
        raise FileNotFoundError("Run `python train.py` before prediction.")
    return latest_model_uri.read_text(encoding="utf-8").strip()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    model_uri = args.model_uri or read_latest_model_uri()
    model = mlflow.pyfunc.load_model(model_uri)
    sample = pd.DataFrame([args.sample], columns=RAW_FEATURE_NAMES)

    print(f"model_uri: {model_uri}")
    print(f"prediction: {model.predict(sample)[0]}")


if __name__ == "__main__":
    main()
