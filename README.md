# Iris MLflow MLOps Project

This project demonstrates a more realistic MLOps loop:

```text
append new raw data -> preprocess -> create data version -> train decision tree -> log MLflow run -> save/register model
```

## Project Structure

```text
iris_mlflow_project/
  data/
    raw/
      iris_events.csv              # growing raw event dataset, generated on first training run
    versions/
      <data_version>/
        train.csv                  # preprocessed train split
        test.csv                   # preprocessed test split
        manifest.json              # data version metadata
  src/
    data.py                        # raw data generation, preprocessing, data versioning
    evaluate.py                    # accuracy and confusion matrix
    model.py                       # pure Python decision tree classifier
  train.py                         # append data + preprocess + train + log to MLflow
  predict.py                       # load latest MLflow model and predict
  mlflow.db                        # MLflow metadata database
  artifacts/                       # MLflow artifacts and logged models
  outputs/
    latest_model_uri.txt
    latest_data_version.txt
```

## Train

Each training run appends new data by default:

```powershell
.venv\Scripts\python.exe train.py
```

Train with a larger data update:

```powershell
.venv\Scripts\python.exe train.py --add-new-data 100 --max-depth 5
```

Do not append data for a debugging rerun:

```powershell
.venv\Scripts\python.exe train.py --add-new-data 0
```

## What Gets Logged to MLflow

Parameters:

```text
model_type
data_version
raw_rows
new_rows_added_this_run
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
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5003 --workers 1
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
