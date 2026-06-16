from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.data import (
    append_new_raw_rows,
    build_features,
    ensure_raw_dataset,
    load_processed_rows,
    preprocess_and_version,
)
from src.evaluate import evaluate
from src.model import FEATURE_NAMES, RAW_FEATURE_NAMES, DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "iris_events.csv"
VERSIONS_DIR = ROOT / "data" / "versions"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"
MODEL_PACKAGE = ROOT / "model_package"
SERVING_SCRIPT = ROOT / "serving" / "predict.py"
SERVING_APP = ROOT / "serving" / "app.py"
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "iris-decision-tree-demo"


class IrisDecisionTreePyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model: DecisionTreeClassifier, medians: dict[str, float]):
        self.model = model
        self.medians = medians

    def predict(self, context, model_input, params=None):  # noqa: ARG002
        rows = []
        for raw_row in model_input[RAW_FEATURE_NAMES].to_dict(orient="records"):
            row = {**raw_row, "label": "unknown", "event_id": "0"}
            rows.append(build_features(row, self.medians))
        return self.model.predict_rows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append data, preprocess, train a decision tree, and log to MLflow.")
    parser.add_argument("--initial-rows", type=int, default=300)
    parser.add_argument("--add-new-data", type=int, default=25)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-split", type=int, default=8)
    parser.add_argument("--register-model-name", default=None)
    parser.add_argument(
        "--raw-data-path",
        default=None,
        help="Optional external raw CSV path mounted by TI-ONE, for example /opt/ml/input/data/iris_events.csv.",
    )
    return parser.parse_args()


def ensure_experiment(tracking_uri: str) -> None:
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def export_model_package(
    model: DecisionTreeClassifier,
    version_info: dict[str, object],
    accuracy: float,
    package_dir: Path = MODEL_PACKAGE,
) -> Path:
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    (package_dir / "decision_tree.json").write_text(
        json.dumps(model.to_dict(), indent=2),
        encoding="utf-8",
    )
    (package_dir / "medians.json").write_text(
        json.dumps(version_info["medians"], indent=2),
        encoding="utf-8",
    )
    (package_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "decision_tree",
                "format": "custom",
                "framework": "python",
                "data_version": version_info["data_version"],
                "raw_rows": version_info["raw_rows"],
                "train_rows": version_info["train_rows"],
                "test_rows": version_info["test_rows"],
                "accuracy": accuracy,
                "input_features": RAW_FEATURE_NAMES,
                "output": "iris_class",
                "entrypoint": "predict.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "requirements.txt").write_text("flask\npandas\n", encoding="utf-8")
    shutil.copy2(SERVING_SCRIPT, package_dir / "predict.py")
    shutil.copy2(SERVING_APP, package_dir / "app.py")
    return package_dir


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    ensure_experiment(tracking_uri)

    if args.raw_data_path:
        external_raw_data = Path(args.raw_data_path)
        if not external_raw_data.exists():
            raise FileNotFoundError(f"raw data file not found: {external_raw_data}")
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(external_raw_data, RAW_DATA_PATH)
        created_rows = 0
    else:
        created_rows = ensure_raw_dataset(RAW_DATA_PATH, args.initial_rows, args.random_state)

    added_rows = append_new_raw_rows(RAW_DATA_PATH, args.add_new_data, args.random_state)
    version_info = preprocess_and_version(RAW_DATA_PATH, VERSIONS_DIR, args.test_ratio, args.random_state)

    train_rows = load_processed_rows(version_info["train_path"])
    test_rows = load_processed_rows(version_info["test_path"])

    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    ).fit(train_rows)
    accuracy, confusion = evaluate(model, test_rows)
    model_package_dir = export_model_package(model, version_info, accuracy)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type": "decision_tree",
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "initial_rows_created": created_rows,
                "new_rows_added_this_run": added_rows,
                "raw_rows": version_info["raw_rows"],
                "train_rows": version_info["train_rows"],
                "test_rows": version_info["test_rows"],
                "missing_values_imputed": version_info["missing_values_imputed"],
                "data_version": version_info["data_version"],
                "test_ratio": args.test_ratio,
                "random_state": args.random_state,
            }
        )
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_dict(model.to_dict(), "decision_tree.json")
        mlflow.log_dict(confusion, "confusion.json")
        mlflow.log_artifact(str(version_info["version_dir"] / "manifest.json"), artifact_path="data")
        mlflow.log_artifact(str(version_info["train_path"]), artifact_path="data")
        mlflow.log_artifact(str(version_info["test_path"]), artifact_path="data")
        mlflow.log_artifacts(str(model_package_dir), artifact_path="model_package")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=IrisDecisionTreePyfuncModel(model, version_info["medians"]),
            input_example=pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=RAW_FEATURE_NAMES),
            registered_model_name=args.register_model_name,
            code_paths=[str(ROOT / "src")],
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        (OUTPUTS / "latest_model_uri.txt").write_text(model_uri, encoding="utf-8")
        (OUTPUTS / "latest_data_version.txt").write_text(str(version_info["data_version"]), encoding="utf-8")

        print(f"experiment: {EXPERIMENT_NAME}")
        print(f"tracking_uri: {tracking_uri}")
        print(f"run_id: {run.info.run_id}")
        print(f"data_version: {version_info['data_version']}")
        print(f"raw_rows: {version_info['raw_rows']} (+{added_rows} this run)")
        print(f"train_rows: {version_info['train_rows']}")
        print(f"test_rows: {version_info['test_rows']}")
        print(f"accuracy: {accuracy:.4f}")
        print(f"model_uri: {model_uri}")
        print(f"model_package_dir: {model_package_dir}")


if __name__ == "__main__":
    main()
