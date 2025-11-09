import os
import sys
import traceback
import argparse
from datetime import datetime
import pandas as pd
import mlflow.pyfunc
import shutil

GCS_BUCKET = "gs://mlops-474118-artifacts"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

def debug(msg: str):
    print(f"[DEBUG] {msg}", flush=True)

# ---------------- Load evaluation CSV ----------------
def fetch_eval_data(path):
    debug(f"Fetching evaluation data from: {path}")
    try:
        if path.startswith("gs://"):
            local_path = "data/tmp_eval.csv"
            os.makedirs("data", exist_ok=True)
            cmd = f"gsutil cp {path} {local_path}"
            debug(f"Running command: {cmd}")
            result = os.system(cmd)
            if result != 0:
                debug(f"[ERROR] gsutil copy failed (exit code {result})")
                sys.exit(1)
            path = local_path
            debug("File successfully copied from GCS.")
        df = pd.read_csv(path)
        debug(f"Successfully read eval CSV with shape: {df.shape}")
        return df
    except Exception as e:
        debug(f"[ERROR] Failed to fetch or read eval data: {e}")
        traceback.print_exc()
        sys.exit(1)

# ---------------- Save predictions to GCS ----------------
def save_preds(df, out_path):
    try:
        df.to_csv("preds.csv", index=False)
        debug(f"File 'preds.csv' saved locally. Uploading to: {out_path}")
        cmd = f"gsutil cp preds.csv {out_path}"
        result = os.system(cmd)
        if result != 0:
            debug(f"[ERROR] Failed to upload preds.csv (exit code {result})")
            sys.exit(1)
        debug("Predictions uploaded to GCS successfully.")
    except Exception as e:
        debug(f"[ERROR] Failed during save_preds: {e}")
        traceback.print_exc()
        sys.exit(1)

# ---------------- Run inference ----------------
def run_inference(best_run_id, eval_csv, local_model_dir=None):
    debug(f"Fetching best model from MLflow run {best_run_id}")
    if local_model_dir:
        debug(f"Loading model from local folder: {local_model_dir}")
        best_model_uri = local_model_dir
    else:
        debug(f"Fetching best model from MLflow run {best_run_id}")
        best_model_uri = f"runs:/{best_run_id}/model"

    try:
        model = mlflow.pyfunc.load_model(best_model_uri)
        debug("Model loaded successfully.")
    except Exception as e:
        debug(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    df = fetch_eval_data(eval_csv)

    # Prepare features
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
    elif 'species' in df.columns:
        X = df.drop(columns=['species'])
    else:
        X = df

    # Run predictions
    try:
        preds = model.predict(X)
        debug(f"Predictions completed. Sample: {preds[:5]}")
    except Exception as e:
        debug(f"[ERROR] Model prediction failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    df['prediction'] = preds
    debug("Predictions column added to DataFrame.")

    # Save predictions to GCS
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = f"{GCS_BUCKET}/inference/{timestamp}"
    save_preds(df, f"{out_dir}/preds.csv")

    # ---------------- Upload best model artifacts to GCS ----------------
    local_model_dir = f"best_model_{timestamp}"
    gcs_model_path = f"{GCS_BUCKET}/models/best_model_{timestamp}"

    debug(f"Downloading MLflow artifacts for run {best_run_id} to {local_model_dir}")
    ret = os.system(f"mlflow artifacts download -r {best_run_id} -d {local_model_dir}")
    if ret != 0:
        debug(f"[ERROR] Failed to download model artifacts (exit code {ret})")
        sys.exit(1)

    debug(f"Uploading best model folder to GCS: {gcs_model_path}")
    ret = os.system(f"gsutil -m cp -r {local_model_dir} {gcs_model_path}")
    if ret != 0:
        debug(f"[ERROR] Failed to upload model to GCS (exit code {ret})")
        sys.exit(1)
    else:
        debug("Best model successfully uploaded to GCS ✅")

    shutil.rmtree(local_model_dir)
    print(f"[SUCCESS] Inference completed. Predictions and best model saved to GCS.", flush=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment_id = "1"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        debug("[ERROR] No runs found in MLflow experiment")
        sys.exit(1)

    best_run_id = runs[0].info.run_id
    print(f"Using best model from run {best_run_id}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_csv", type=str, default=f"{GCS_BUCKET}/data/processed/eval.csv",
        help="Path to evaluation CSV (local or GCS)")
    parser.add_argument(
        "--local_model_dir", type=str, default=None,
        help="Optional local model folder to use instead of MLflow download"
    )
    args = parser.parse_args()

    try:
        run_inference(best_run_id, args.eval_csv)
        debug("Inference script completed successfully ✅")
    except Exception as e:
        debug(f"[FATAL] Inference script failed: {e}")
        traceback.print_exc()
        sys.exit(1)
