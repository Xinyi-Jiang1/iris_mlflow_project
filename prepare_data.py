from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from src.data import append_new_raw_rows, ensure_raw_dataset, preprocess_and_version


ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "iris_events.csv"
VERSIONS_DIR = ROOT / "data" / "versions"
RUNTIME = ROOT / "_runtime"
WANDB_RUNTIME = RUNTIME / "wandb"
OUTPUTS = RUNTIME / "outputs"
DEFAULT_WANDB_PROJECT = "iris-mlflow-project"
WANDB_DATASET_NAME = "iris-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append raw data, preprocess it, and create a versioned dataset.")
    parser.add_argument("--initial-rows", type=int, default=300)
    parser.add_argument("--add-new-data", type=int, default=25)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log the prepared dataset as a W&B dataset artifact output.",
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
    parser.add_argument(
        "--raw-data-path",
        default=None,
        help="Optional external raw CSV path mounted by TI-ONE, for example /opt/ml/input/data/iris_events.csv.",
    )
    return parser.parse_args()


def init_wandb_run(args: argparse.Namespace, version_info: dict[str, object], created_rows: int, added_rows: int):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging was requested, but wandb is not installed. Run: pip install wandb") from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": f"prepare-iris-data-{version_info['data_version']}",
        "job_type": "data-prep",
        "dir": str(WANDB_RUNTIME),
        "config": {
            "data_version": version_info["data_version"],
            "initial_rows_created": created_rows,
            "new_rows_added_this_run": added_rows,
            "raw_rows": version_info["raw_rows"],
            "train_rows": version_info["train_rows"],
            "test_rows": version_info["test_rows"],
            "missing_values_imputed": version_info["missing_values_imputed"],
            "test_ratio": args.test_ratio,
            "random_state": args.random_state,
        },
    }
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode

    return wandb.init(**init_kwargs)


def log_wandb_dataset(wandb_run, version_info: dict[str, object]) -> None:
    if wandb_run is None:
        return

    import wandb

    artifact = wandb.Artifact(
        name=WANDB_DATASET_NAME,
        type="dataset",
        metadata={
            "data_version": version_info["data_version"],
            "raw_rows": version_info["raw_rows"],
            "train_rows": version_info["train_rows"],
            "test_rows": version_info["test_rows"],
            "missing_values_imputed": version_info["missing_values_imputed"],
        },
    )
    artifact.add_file(str(version_info["version_dir"] / "manifest.json"), name="manifest.json")
    artifact.add_file(str(version_info["train_path"]), name="train.csv")
    artifact.add_file(str(version_info["test_path"]), name="test.csv")
    wandb_run.log_artifact(artifact, aliases=["latest", f"data-{version_info['data_version']}"])


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    WANDB_RUNTIME.mkdir(parents=True, exist_ok=True)

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

    (OUTPUTS / "latest_data_version.txt").write_text(str(version_info["data_version"]), encoding="utf-8")
    (OUTPUTS / "latest_dataset_manifest.json").write_text(
        json.dumps(
            {
                "data_version": version_info["data_version"],
                "version_dir": str(version_info["version_dir"]),
                "train_path": str(version_info["train_path"]),
                "test_path": str(version_info["test_path"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    wandb_run = init_wandb_run(args, version_info, created_rows, added_rows)
    try:
        log_wandb_dataset(wandb_run, version_info)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"data_version: {version_info['data_version']}")
    print(f"raw_rows: {version_info['raw_rows']} (+{added_rows} this run)")
    print(f"train_rows: {version_info['train_rows']}")
    print(f"test_rows: {version_info['test_rows']}")
    print(f"version_dir: {version_info['version_dir']}")


if __name__ == "__main__":
    main()
