from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RAW_FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_package(package_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    tree = json.loads((package_dir / "decision_tree.json").read_text(encoding="utf-8"))
    medians = json.loads((package_dir / "medians.json").read_text(encoding="utf-8"))
    return tree, {name: float(value) for name, value in medians.items()}


def build_features(raw_row: dict[str, Any], medians: dict[str, float]) -> dict[str, float]:
    values = {}
    for name in RAW_FEATURE_NAMES:
        raw_value = raw_row.get(name, "")
        values[name] = float(raw_value) if raw_value != "" else medians[name]

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
    }


def predict_one(tree: dict[str, Any], features: dict[str, float]) -> str:
    node = tree
    while "feature" in node:
        branch = "left" if features[node["feature"]] <= float(node["threshold"]) else "right"
        node = node[branch]
    return str(node["prediction"])


def parse_payload(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    elif args.json:
        payload = json.loads(args.json)
    else:
        payload = {
            "sepal_length": args.sample[0],
            "sepal_width": args.sample[1],
            "petal_length": args.sample[2],
            "petal_width": args.sample[3],
        }

    if isinstance(payload, dict) and "instances" in payload:
        return list(payload["instances"])
    if isinstance(payload, list):
        return payload
    return [payload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Iris classes from an exported model package.")
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--sample", nargs=4, type=float, default=[5.1, 3.5, 1.4, 0.2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tree, medians = load_package(Path(args.model_dir))
    rows = parse_payload(args)
    predictions = [predict_one(tree, build_features(row, medians)) for row in rows]
    print(json.dumps({"predictions": predictions}, ensure_ascii=True))


if __name__ == "__main__":
    main()
