from __future__ import annotations

from collections import defaultdict

from src.model import DecisionTreeClassifier


def evaluate(
    model: DecisionTreeClassifier,
    rows: list[dict[str, str]],
) -> tuple[float, dict[str, int]]:
    correct = 0
    confusion = defaultdict(int)

    for row in rows:
        actual = row["label"]
        predicted = model.predict_one(row)
        correct += int(predicted == actual)
        confusion[f"{actual} predicted_as {predicted}"] += 1

    return correct / len(rows), dict(confusion)
