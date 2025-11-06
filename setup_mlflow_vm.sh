#!/bin/bash
# Setup MLflow tracking server on GCP Compute Engine
PROJECT_ID="mlops-474118"
REGION="us-central1"
BUCKET_NAME="mlops-474118-artifacts"
ZONE="us-central1-a"
VM_NAME="mlflow-tracking-server"

echo "Creating VM for MLflow in project: $PROJECT_ID ($ZONE)..."
gcloud compute instances create $VM_NAME   --project=$PROJECT_ID   --zone=$ZONE   --machine-type=e2-medium   --image-family=debian-11   --image-project=debian-cloud   --boot-disk-size=30GB   --tags=http-server,https-server   --metadata startup-script='#! /bin/bash
  apt update && apt install -y python3-pip
  pip3 install mlflow google-cloud-storage gunicorn
  mkdir /opt/mlflow
  export MLFLOW_ARTIFACT_ROOT=gs://'"$BUCKET_NAME"'/mlflow
  nohup mlflow server       --backend-store-uri sqlite:///mlflow.db       --default-artifact-root $MLFLOW_ARTIFACT_ROOT       --host 0.0.0.0 --port 5000 > /opt/mlflow/mlflow.log 2>&1 &
  '

echo "VM create issued. To get the external IP:"
gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
echo "After the VM is running, update mlflow_config.yaml with the returned IP."
