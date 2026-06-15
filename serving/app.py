from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request

from predict import build_features, load_package, predict_one


MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent))
PORT = int(os.environ.get("PORT", os.environ.get("SERVICE_PORT", "8080")))

tree, medians = load_package(MODEL_DIR)
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True, silent=False)
    rows = payload.get("instances", payload if isinstance(payload, list) else [payload])
    predictions = [predict_one(tree, build_features(row, medians)) for row in rows]
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
