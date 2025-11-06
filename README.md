# IRIS MLOps (GCP-ready)

This is a ready-to-run starter repository for an **end-to-end MLOps pipeline** for the IRIS dataset — fully configured to run **on Google Cloud Platform (GCP)** in project **mlops-474118** and region **us-central1**.

## What's included
- GCP-first scripts and configs (no local references)
- DVC remote configured to use GCS
- Feast feature store config (BigQuery offline, SQLite online)
- MLflow deployment script for a Compute Engine VM
- Vertex AI compatible pipeline script and training/inference scripts
- GitHub Actions CI workflow with CML reporting
- Tests (pytest) for data validation & model eval

## Important placeholders to update
1. Replace `<MLFLOW_SERVER_IP>` in `mlflow_config.yaml` after you run `setup_mlflow_vm.sh`.
2. Set your GitHub repo remote and push to `lekshmisuseela/iris-mlops`.

## Quick start (on Vertex AI Workbench)
1. Clone repo to your Vertex Workbench VM / JupyterLab.
2. Install deps: `pip install -r requirements.txt`
3. Create bucket: `gsutil mb -p mlops-474118 -l us-central1 gs://mlops-474118-artifacts/`
4. Run MLflow VM setup: `bash setup_mlflow_vm.sh`
5. Update `mlflow_config.yaml` with MLflow tracking URI
6. Initialize DVC and push data to GCS (see README docs in repo)

---
Project: mlops-474118 (GCP)
Bucket: gs://mlops-474118-artifacts
Region: us-central1
GitHub: lekshmisuseela
