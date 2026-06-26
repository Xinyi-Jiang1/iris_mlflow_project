# GitLab MLflow Backend Setup

This project can use GitLab as the MLflow Tracking and Model Registry backend.

## 1. Configure credentials

Create a GitLab personal, project, or group access token with the `api` scope and Developer or higher access.

PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="https://<gitlab-host>/api/v4/projects/<project-id>/ml/mlflow"
$env:MLFLOW_TRACKING_TOKEN="<gitlab-access-token>"
```

The same keys are listed in `.env.example`. Do not commit a real token.

## 2. Prepare data

```powershell
.venv\Scripts\python.exe prepare_data.py --add-new-data 0
```

## 3. Log an experiment run to GitLab

```powershell
.venv\Scripts\python.exe train.py
```

When `MLFLOW_TRACKING_URI` points at `/api/v4/projects/<project-id>/ml/mlflow`, `train.py` switches to GitLab-compatible artifact logging. In GitLab, the run appears under `Analyze > Model experiments`.

## 4. Register a model version

GitLab Model Registry versions must use semantic versioning.

```powershell
.venv\Scripts\python.exe train.py --register-model-name iris-decision-tree --gitlab-model-version 0.1.0
```

If `--gitlab-model-version` is omitted, the script creates a timestamp patch version such as `0.1.20260626123045`.

Registered models appear under `Deploy > Model registry`.

## 5. GitLab CI/CD variables

Store these as CI/CD variables:

```text
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_TOKEN
```

The training script tags runs with these GitLab CI values when present:

```text
gitlab.CI_JOB_ID
git.commit
git.branch
```
