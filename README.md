# Iris MLflow MLOps Project

This project demonstrates a more realistic MLOps loop:

```text
prepare raw data -> create dataset artifact -> train decision tree -> log MLflow run -> log model artifact
```

## Project Structure

```text
iris_mlflow_project/
  prepare_data.py                  # data prep entry point: creates data versions and optional W&B dataset artifacts
  train.py                         # training entry point: consumes a prepared data version, logs to MLflow and optionally W&B
  predict.py                       # loads latest MLflow model URI and predicts
  requirements.txt
  README.md

  src/                             # reusable data, model, and evaluation code
  serving/                         # files copied into the exported serving package

  data/
    raw/
      iris_events.csv              # shared raw training data source
    versions/                      # shared processed train/test versions generated from raw data
      <data_version>/
        manifest.json
        train.csv
        test.csv

  _runtime/                        # generated local runtime state, ignored by Git
    mlflow/
      mlflow.db                    # MLflow metadata DB
      artifacts/                   # MLflow artifact store and logged MLflow models
    wandb/                         # W&B local run/cache files, including offline runs
    outputs/
      latest_model_uri.txt         # pointer used by predict.py
      latest_data_version.txt
    model_package/                 # exported model files before logging to MLflow/W&B
```

## MLflow vs W&B Files

Shared by both platforms:

```text
data/raw/iris_events.csv
data/versions/<data_version>/manifest.json
data/versions/<data_version>/train.csv
data/versions/<data_version>/test.csv
_runtime/model_package/
```

MLflow-only local state:

```text
_runtime/mlflow/mlflow.db
_runtime/mlflow/artifacts/
_runtime/outputs/latest_model_uri.txt
```

W&B-only local state:

```text
_runtime/wandb/
```

The important idea is: MLflow and W&B observe the same training run, same dataset version, and same exported model package. They do not share a database. MLflow stores its own tracking database and artifact store locally; W&B stores local cache/offline files in `_runtime/wandb/` and uploads run/artifact metadata to W&B Cloud when online mode is used.

## Prepare Data

Create or update the raw dataset, preprocess it, and write a versioned train/test split:

```powershell
.venv\Scripts\python.exe prepare_data.py
```

Prepare data without appending new raw rows:

```powershell
.venv\Scripts\python.exe prepare_data.py --add-new-data 0
```

Prepare data with W&B enabled. This creates a W&B `data-prep` run and logs `iris-dataset` as that run's output artifact:

```powershell
.venv\Scripts\python.exe prepare_data.py --wandb --wandb-project iris-mlflow-project
```

## Train

Train from the latest prepared data version and log the model to MLflow:

```powershell
.venv\Scripts\python.exe train.py
```

Train from a specific prepared data version:

```powershell
.venv\Scripts\python.exe train.py --data-version 7ad0c4f025ac
```

Train with W&B enabled. In online mode, this consumes `iris-dataset:data-<data_version>` and logs `iris-decision-tree-model` as the output artifact:

```powershell
.venv\Scripts\python.exe prepare_data.py --wandb --wandb-project iris-mlflow-project
.venv\Scripts\python.exe train.py --wandb --wandb-project iris-mlflow-project
```

## Optional W&B Lineage Logging

MLflow remains the main local tracking and registry layer. To also send the same training run to Weights & Biases for richer charts and artifact lineage, install dependencies and log in once:

```powershell
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\wandb.exe login
```

Run the split W&B lineage flow:

```powershell
.venv\Scripts\python.exe prepare_data.py --wandb --wandb-project iris-mlflow-project
.venv\Scripts\python.exe train.py --wandb --wandb-project iris-mlflow-project
```

For a local dry run that writes W&B files without uploading. W&B offline mode cannot declare input artifacts, so this validates logging locally but the full data-prep -> dataset -> train -> model graph appears after an online run:

```powershell
.venv\Scripts\python.exe prepare_data.py --wandb --wandb-mode offline --add-new-data 0
.venv\Scripts\python.exe train.py --wandb --wandb-mode offline
```

When `--wandb` is enabled, the run logs:

```text
accuracy metric
training config, including the MLflow run id
prepare_data.py output dataset artifact: iris-dataset
train.py input dataset artifact: iris-dataset:data-<data_version>
train.py output model artifact: iris-decision-tree-model
```

This keeps MLflow as the source for model registration/deployment while W&B shows the data-prep run -> dataset artifact -> training run -> model artifact relationship.

## What Gets Logged to MLflow

Parameters:

```text
model_type
data_version
raw_rows
train_rows
test_rows
missing_values_imputed
max_depth
min_samples_split
```

Metrics:

```text
accuracy
```

Artifacts:

```text
decision_tree.json
confusion.json
data/manifest.json
data/train.csv
data/test.csv
model artifacts
```

## Predict

```powershell
.venv\Scripts\python.exe predict.py
```

## Open MLflow UI

```powershell
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///_runtime/mlflow/mlflow.db --port 5003 --workers 1
```

Open:

```text
http://127.0.0.1:5003
```

## Why Data Version Matters

Every time raw data changes, preprocessing produces a new `data_version`. MLflow logs that version with the run, so you can answer:

```text
Which dataset produced this model?
Which preprocessing config was used?
How many new rows were added?
Can another person rerun this training with the same data split?
```
