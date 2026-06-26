from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.data import build_features, load_processed_rows
from src.evaluate import evaluate
from src.model import FEATURE_NAMES, RAW_FEATURE_NAMES, DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "iris_events.csv"
VERSIONS_DIR = ROOT / "data" / "versions"
RUNTIME = ROOT / "_runtime"
MLFLOW_RUNTIME = RUNTIME / "mlflow"
WANDB_RUNTIME = RUNTIME / "wandb"
OUTPUTS = RUNTIME / "outputs"
ARTIFACTS = MLFLOW_RUNTIME / "artifacts"
MODEL_PACKAGE = RUNTIME / "model_package"
SERVING_SCRIPT = ROOT / "serving" / "predict.py"
SERVING_APP = ROOT / "serving" / "app.py"
DEFAULT_TRACKING_URI = f"sqlite:///{(MLFLOW_RUNTIME / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "iris-decision-tree-demo"
DEFAULT_WANDB_PROJECT = "iris-mlflow-project"
WANDB_DATASET_NAME = "iris-dataset"


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
    parser = argparse.ArgumentParser(description="Train a decision tree from a prepared data version and log to MLflow.")
    parser.add_argument("--data-version", default=None, help="Prepared data version to train from. Defaults to _runtime/outputs/latest_data_version.txt.")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-split", type=int, default=8)
    parser.add_argument("--register-model-name", default=None)
    parser.add_argument(
        "--gitlab-model-version",
        default=os.environ.get("GITLAB_MODEL_VERSION"),
        help="Semantic version to create in GitLab Model Registry. Defaults to a timestamp patch version.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Also log this run to Weights & Biases for experiment visualization and artifact lineage.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        help=f"W&B project name. Defaults to WANDB_PROJECT or {DEFAULT_WANDB_PROJECT}.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Optional W&B team/user entity. Defaults to WANDB_ENTITY.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE"),
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Use offline to test logging without uploading.",
    )
    return parser.parse_args()


def ensure_experiment(tracking_uri: str) -> None:
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        if is_gitlab_tracking_uri(tracking_uri):
            client.create_experiment(EXPERIMENT_NAME)
        else:
            client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def is_gitlab_tracking_uri(tracking_uri: str) -> bool:
    return os.environ.get("GITLAB_MLFLOW") == "1" or (
        "/api/v4/projects/" in tracking_uri and tracking_uri.endswith("/ml/mlflow")
    )


def next_gitlab_model_version(explicit_version: str | None) -> str:
    if explicit_version:
        return explicit_version
    patch = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"0.1.{patch}"


def read_latest_data_version() -> str:
    latest_data_version = OUTPUTS / "latest_data_version.txt"
    if not latest_data_version.exists():
        raise FileNotFoundError("Run `python prepare_data.py` before training, or pass --data-version.")
    return latest_data_version.read_text(encoding="utf-8").strip()


def load_version_info(data_version: str) -> dict[str, object]:
    version_dir = VERSIONS_DIR / data_version
    manifest_path = version_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"data version not found: {data_version}. Expected {manifest_path}")
    version_info = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        **version_info,
        "version_dir": version_dir,
        "train_path": version_dir / "train.csv",
        "test_path": version_dir / "test.csv",
    }


def init_wandb_run(args: argparse.Namespace, params: dict[str, object], mlflow_run_id: str):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging was requested, but wandb is not installed. Run: pip install wandb") from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": f"iris-{params['data_version']}-{mlflow_run_id[:8]}",
        "job_type": "train",
        "dir": str(WANDB_RUNTIME),
        "config": {
            **params,
            "mlflow_experiment": EXPERIMENT_NAME,
            "mlflow_run_id": mlflow_run_id,
        },
    }
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode

    return wandb.init(**init_kwargs)


def log_wandb_artifacts(wandb_run, version_info: dict[str, object], model_package_dir: Path, accuracy: float) -> None:
    if wandb_run is None:
        return

    import wandb

    if getattr(wandb_run.settings, "mode", None) != "offline":
        wandb_run.use_artifact(f"{WANDB_DATASET_NAME}:data-{version_info['data_version']}")

    model_artifact = wandb.Artifact(
        name="iris-decision-tree-model",
        type="model",
        metadata={
            "model_type": "decision_tree",
            "data_version": version_info["data_version"],
            "accuracy": accuracy,
        },
    )
    model_artifact.add_dir(str(model_package_dir))
    wandb_run.log_artifact(model_artifact, aliases=["latest", f"data-{version_info['data_version']}"])


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


def log_gitlab_flat_artifacts(version_info: dict[str, object], model_package_dir: Path) -> None:
    """GitLab's MLflow endpoint supports artifacts best as flat files at the run root."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copy2(version_info["version_dir"] / "manifest.json", temp_path / "data_manifest.json")
        shutil.copy2(version_info["train_path"], temp_path / "data_train.csv")
        shutil.copy2(version_info["test_path"], temp_path / "data_test.csv")

        for source_path in sorted(model_package_dir.iterdir()):
            if source_path.is_file():
                shutil.copy2(source_path, temp_path / f"model_package_{source_path.name}")

        for artifact_path in sorted(temp_path.iterdir()):
            mlflow.log_artifact(str(artifact_path), artifact_path="")


def create_or_update_gitlab_model_version(
    client: mlflow.MlflowClient,
    model_name: str,
    model_version: str,
    params: dict[str, object],
    accuracy: float,
    model_package_dir: Path,
) -> None:
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name, description="Iris decision tree model tracked from MLflow.")

    try:
        version = client.create_model_version(
            model_name,
            source="",
            description=f"Iris decision tree trained from data version {params['data_version']}.",
            tags={"gitlab.version": model_version},
        )
    except Exception:
        version = client.get_model_version(model_name, model_version)

    run_id = version.run_id
    client.log_param(run_id, "model_type", params["model_type"])
    client.log_param(run_id, "data_version", params["data_version"])
    client.log_param(run_id, "max_depth", params["max_depth"])
    client.log_param(run_id, "min_samples_split", params["min_samples_split"])
    client.log_metric(run_id, "accuracy", accuracy)

    if os.getenv("GITLAB_CI"):
        client.set_tag(run_id, "gitlab.CI_JOB_ID", os.getenv("CI_JOB_ID"))

    for artifact_path in sorted(model_package_dir.iterdir()):
        if artifact_path.is_file():
            client.log_artifact(run_id, str(artifact_path), artifact_path="")


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    WANDB_RUNTIME.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    ensure_experiment(tracking_uri)
    gitlab_tracking = is_gitlab_tracking_uri(tracking_uri)

    data_version = args.data_version or read_latest_data_version()
    version_info = load_version_info(data_version)

    train_rows = load_processed_rows(version_info["train_path"])
    test_rows = load_processed_rows(version_info["test_path"])

    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    ).fit(train_rows)
    accuracy, confusion = evaluate(model, test_rows)
    model_package_dir = export_model_package(model, version_info, accuracy)
    params = {
        "model_type": "decision_tree",
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "raw_rows": version_info["raw_rows"],
        "train_rows": version_info["train_rows"],
        "test_rows": version_info["test_rows"],
        "missing_values_imputed": version_info["missing_values_imputed"],
        "data_version": version_info["data_version"],
        "test_ratio": version_info["preprocess_config"]["test_ratio"],
        "random_state": version_info["preprocess_config"]["random_state"],
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        if os.getenv("GITLAB_CI"):
            mlflow.set_tag("gitlab.CI_JOB_ID", os.getenv("CI_JOB_ID"))
        if os.getenv("CI_COMMIT_SHA"):
            mlflow.set_tag("git.commit", os.getenv("CI_COMMIT_SHA"))
        if os.getenv("CI_COMMIT_REF_NAME"):
            mlflow.set_tag("git.branch", os.getenv("CI_COMMIT_REF_NAME"))

        mlflow.log_dict(model.to_dict(), "decision_tree.json")
        mlflow.log_dict(confusion, "confusion.json")
        if gitlab_tracking:
            log_gitlab_flat_artifacts(version_info, model_package_dir)
        else:
            mlflow.log_artifact(str(version_info["version_dir"] / "manifest.json"), artifact_path="data")
            mlflow.log_artifact(str(version_info["train_path"]), artifact_path="data")
            mlflow.log_artifact(str(version_info["test_path"]), artifact_path="data")
            mlflow.log_artifacts(str(model_package_dir), artifact_path="model_package")
        if gitlab_tracking:
            # GitLab stores the exported serving package as run artifacts. Avoid
            # mlflow.pyfunc.log_model() here because recent MLflow clients call
            # logged-model APIs that GitLab's compatibility layer may reject.
            model_uri = f"runs:/{run.info.run_id}/"
        else:
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

        if gitlab_tracking and args.register_model_name:
            client = mlflow.MlflowClient(tracking_uri=tracking_uri)
            model_version = next_gitlab_model_version(args.gitlab_model_version)
            create_or_update_gitlab_model_version(
                client,
                args.register_model_name,
                model_version,
                params,
                accuracy,
                model_package_dir,
            )

        wandb_run = init_wandb_run(args, params, run.info.run_id)
        try:
            if wandb_run is not None:
                wandb_run.log({"accuracy": accuracy})
                log_wandb_artifacts(wandb_run, version_info, model_package_dir, accuracy)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        print(f"experiment: {EXPERIMENT_NAME}")
        print(f"tracking_uri: {tracking_uri}")
        print(f"gitlab_tracking: {gitlab_tracking}")
        print(f"run_id: {run.info.run_id}")
        if gitlab_tracking and args.register_model_name:
            print(f"registered_model_name: {args.register_model_name}")
            print(f"gitlab_model_version: {model_version}")
        print(f"data_version: {version_info['data_version']}")
        print(f"raw_rows: {version_info['raw_rows']}")
        print(f"train_rows: {version_info['train_rows']}")
        print(f"test_rows: {version_info['test_rows']}")
        print(f"accuracy: {accuracy:.4f}")
        print(f"model_uri: {model_uri}")
        print(f"model_package_dir: {model_package_dir}")


if __name__ == "__main__":
    main()
