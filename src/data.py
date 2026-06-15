from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

from src.model import FEATURE_NAMES, RAW_FEATURE_NAMES

LABEL_PROFILES = {
    "setosa": [5.0, 3.4, 1.5, 0.25],
    "versicolor": [6.1, 2.8, 4.4, 1.35],
    "virginica": [6.6, 3.0, 5.6, 2.05],
}

RAW_COLUMNS = [*RAW_FEATURE_NAMES, "label", "event_id", "event_date"]


def ensure_raw_dataset(path: Path, initial_rows: int, random_state: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return 0
    rows = generate_raw_rows(initial_rows, start_id=1, random_state=random_state)
    write_rows(path, rows)
    return len(rows)


def append_new_raw_rows(path: Path, count: int, random_state: int) -> int:
    if count <= 0:
        return 0
    existing = load_rows(path) if path.exists() else []
    start_id = len(existing) + 1
    rows = generate_raw_rows(count, start_id=start_id, random_state=random_state + start_id)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def generate_raw_rows(count: int, start_id: int, random_state: int) -> list[dict[str, str]]:
    rng = random.Random(random_state)
    labels = list(LABEL_PROFILES)
    rows = []
    for offset in range(count):
        label = labels[(start_id + offset) % len(labels)]
        base = LABEL_PROFILES[label]
        features = [round(value + rng.gauss(0, noise), 3) for value, noise in zip(base, [0.28, 0.2, 0.32, 0.16])]
        features = [max(value, 0.05) for value in features]

        row = {
            name: str(value)
            for name, value in zip(RAW_FEATURE_NAMES, features)
        }
        row.update(
            {
                "label": label,
                "event_id": str(start_id + offset),
                "event_date": f"2026-06-{1 + ((start_id + offset) % 28):02d}",
            }
        )

        # Keep a tiny bit of realistic mess so preprocessing has visible work.
        if rng.random() < 0.035:
            row[rng.choice(RAW_FEATURE_NAMES)] = ""
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preprocess_and_version(
    raw_path: Path,
    versions_dir: Path,
    test_ratio: float,
    random_state: int,
) -> dict[str, object]:
    raw_rows = load_rows(raw_path)
    medians = compute_medians(raw_rows)
    processed = [build_features(row, medians) for row in raw_rows]
    processed.sort(key=lambda row: int(row["event_id"]))

    config = {
        "test_ratio": test_ratio,
        "random_state": random_state,
        "features": FEATURE_NAMES,
        "imputation": "median",
        "feature_engineering": ["petal_ratio", "sepal_ratio", "petal_area", "sepal_area"],
    }
    data_version = compute_data_version(processed, config)
    version_dir = versions_dir / data_version

    rng = random.Random(random_state)
    shuffled = list(processed)
    rng.shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    test_rows = shuffled[:test_size]
    train_rows = shuffled[test_size:]

    version_dir.mkdir(parents=True, exist_ok=True)
    write_processed_csv(version_dir / "train.csv", train_rows)
    write_processed_csv(version_dir / "test.csv", test_rows)

    manifest = {
        "data_version": data_version,
        "raw_path": str(raw_path),
        "raw_rows": len(raw_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "missing_values_imputed": count_missing(raw_rows),
        "medians": medians,
        "preprocess_config": config,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "version_dir": version_dir, "train_path": version_dir / "train.csv", "test_path": version_dir / "test.csv"}


def compute_medians(rows: list[dict[str, str]]) -> dict[str, float]:
    medians = {}
    for name in RAW_FEATURE_NAMES:
        values = sorted(float(row[name]) for row in rows if row.get(name))
        middle = len(values) // 2
        medians[name] = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return medians


def build_features(row: dict[str, str], medians: dict[str, float] | None = None) -> dict[str, float | str]:
    medians = medians or {name: float(row[name]) for name in RAW_FEATURE_NAMES if row.get(name)}
    values = {}
    for name in RAW_FEATURE_NAMES:
        raw_value = row.get(name, "")
        values[name] = float(raw_value) if raw_value != "" else float(medians[name])

    sepal_length = values["sepal_length"]
    sepal_width = values["sepal_width"]
    petal_length = values["petal_length"]
    petal_width = values["petal_width"]

    return {
        **values,
        "petal_ratio": petal_length / max(petal_width, 1e-6),
        "sepal_ratio": sepal_length / max(sepal_width, 1e-6),
        "petal_area": petal_length * petal_width,
        "sepal_area": sepal_length * sepal_width,
        "label": row["label"],
        "event_id": row.get("event_id", "0"),
    }


def write_processed_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [*FEATURE_NAMES, "label", "event_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_processed_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute_data_version(rows: list[dict[str, float | str]], config: dict[str, object]) -> str:
    payload = json.dumps({"rows": rows, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def count_missing(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows for name in RAW_FEATURE_NAMES if row.get(name, "") == "")
