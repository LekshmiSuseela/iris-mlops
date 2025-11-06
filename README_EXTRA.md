## MLflow VM setup
Run `bash setup_mlflow_vm.sh` from a GCP console (or Vertex Workbench) authenticated with gcloud.
The script creates a Compute Engine VM and starts mlflow server with artifact root pointing to GCS.
After the VM is created, run:
```
gcloud compute instances describe mlflow-tracking-server --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```
Take the returned IP and set `mlflow_config.yaml` `tracking_uri` to `http://<IP>:5000`.

## DVC remote
The starter `dvc.yaml` assumes a remote `gcsremote`. Configure it in Vertex Workbench:
```
dvc remote add -d gcsremote gs://mlops-474118-artifacts/dvcstore
dvc remote modify gcsremote credentialpath /path/to/sa.json
```

## Feast
Feast configuration is in `feast_repo/feature_store.yaml`. It uses BigQuery dataset `iris_features`.
