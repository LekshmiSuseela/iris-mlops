import argparse
import subprocess
import os
import sys
import traceback
from datetime import datetime


def debug(msg: str):
    """Standardized debug print with timestamp."""
    print(f"[{datetime.utcnow().isoformat()}] [DEBUG] {msg}", flush=True)


def run_pipeline(data_version='v1'):
    debug(f"Starting pipeline for data_version={data_version}")

    # Environment setup
    try:
        env = os.environ.copy()

        # Select dataset version
        if data_version == 'v2':
            env['DATA_CSV'] = "gs://mlops-474118-artifacts/data/iris_v2.csv"
            debug("Using augmented dataset: gs://mlops-474118-artifacts/data/iris_v2.csv")
        else:
            env['DATA_CSV'] = "gs://mlops-474118-artifacts/data/iris.csv"
            debug("Using base dataset: gs://mlops-474118-artifacts/data/iris.csv")

        # MLflow tracking
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://<MLFLOW_SERVER_IP>:5000')
        env['MLFLOW_TRACKING_URI'] = mlflow_uri
        debug(f"MLflow tracking URI set to: {mlflow_uri}")

        # Optional sanity check
        debug(f"Environment variables set: DATA_CSV={env['DATA_CSV']}, MLFLOW_TRACKING_URI={mlflow_uri}")
    except Exception as e:
        debug(f"[ERROR] Failed to configure environment: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 1: Train model
    try:
        debug("Running training script (scripts/train.py)...")
        subprocess.check_call(['python', 'scripts/train.py'], env=env)
        debug("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        debug(f"[ERROR] Training script failed with exit code {e.returncode}")
        traceback.print_exc()
        sys.exit(e.returncode)
    except Exception as e:
        debug(f"[ERROR] Unexpected error during training: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Inference
    try:
        debug("Running inference script (scripts/inference.py)...")
        subprocess.check_call(['python', 'scripts/inference.py'], env=env)
        debug("Inference completed successfully.")
    except subprocess.CalledProcessError as e:
        debug(f"[ERROR] Inference script failed with exit code {e.returncode}")
        traceback.print_exc()
        sys.exit(e.returncode)
    except Exception as e:
        debug(f"[ERROR] Unexpected error during inference: {e}")
        traceback.print_exc()
        sys.exit(1)

    debug("Pipeline finished successfully ✅")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training + inference pipeline on GCP")
    parser.add_argument('--data_version', type=str, default='v1', help="Dataset version (v1 or v2)")
    args = parser.parse_args()

    try:
        run_pipeline(args.data_version)
    except Exception as e:
        debug(f"[FATAL] Pipeline crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
