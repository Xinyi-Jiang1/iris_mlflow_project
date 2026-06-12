from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

RAW_FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "petal_ratio",
    "sepal_ratio",
    "petal_area",
    "sepal_area",
]


@dataclass
class DecisionNode:
    prediction: str
    feature: str | None = None
    threshold: float | None = None
    left: DecisionNode | None = None
    right: DecisionNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 4, min_samples_split: int = 8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: DecisionNode | None = None

    def fit(self, rows: list[dict[str, str]]) -> DecisionTreeClassifier:
        self.root = self._build(rows, depth=0)
        return self

    def predict_rows(self, rows: list[dict[str, Any]]) -> list[str]:
        return [self.predict_one(row) for row in rows]

    def predict_one(self, row: dict[str, Any]) -> str:
        if self.root is None:
            raise ValueError("Model has not been fitted")
        node = self.root
        while not node.is_leaf:
            value = float(row[node.feature])
            node = node.left if value <= node.threshold else node.right
        return node.prediction

    def to_dict(self) -> dict[str, Any]:
        if self.root is None:
            return {}
        return self._node_to_dict(self.root)

    def _build(self, rows: list[dict[str, str]], depth: int) -> DecisionNode:
        prediction = majority_label(rows)
        labels = {row["label"] for row in rows}
        if depth >= self.max_depth or len(rows) < self.min_samples_split or len(labels) == 1:
            return DecisionNode(prediction=prediction)

        split = best_split(rows)
        if split is None:
            return DecisionNode(prediction=prediction)

        feature, threshold, left_rows, right_rows = split
        return DecisionNode(
            prediction=prediction,
            feature=feature,
            threshold=threshold,
            left=self._build(left_rows, depth + 1),
            right=self._build(right_rows, depth + 1),
        )

    def _node_to_dict(self, node: DecisionNode) -> dict[str, Any]:
        if node.is_leaf:
            return {"prediction": node.prediction}
        return {
            "feature": node.feature,
            "threshold": node.threshold,
            "prediction": node.prediction,
            "left": self._node_to_dict(node.left),
            "right": self._node_to_dict(node.right),
        }


def majority_label(rows: list[dict[str, str]]) -> str:
    return Counter(row["label"] for row in rows).most_common(1)[0][0]


def gini(rows: list[dict[str, str]]) -> float:
    total = len(rows)
    counts = Counter(row["label"] for row in rows)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def best_split(rows: list[dict[str, str]]) -> tuple[str, float, list[dict[str, str]], list[dict[str, str]]] | None:
    parent_gini = gini(rows)
    best_gain = 0.0
    best: tuple[str, float, list[dict[str, str]], list[dict[str, str]]] | None = None

    for feature in FEATURE_NAMES:
        values = sorted({float(row[feature]) for row in rows})
        thresholds = [(left + right) / 2 for left, right in zip(values, values[1:], strict=False)]
        for threshold in thresholds:
            left_rows = [row for row in rows if float(row[feature]) <= threshold]
            right_rows = [row for row in rows if float(row[feature]) > threshold]
            if not left_rows or not right_rows:
                continue
            weighted = (len(left_rows) / len(rows)) * gini(left_rows) + (len(right_rows) / len(rows)) * gini(right_rows)
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain = gain
                best = (feature, threshold, left_rows, right_rows)

    return best
