import os
import sys
import traceback
import joblib
import argparse
from datetime import datetime
import pandas as pd
import mlflow.pyfunc
import shutil

GCS_BUCKET = "gs://mlops-474118-artifacts"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

def debug(msg: str):
    """Unified debug logger."""
    print(f"[DEBUG] {msg}", flush=True)

# ---------------- Data Loading ----------------
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
                debug(f"[ERROR] gsutil copy failed with exit code {result}")
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

# ---------------- Save predictions ----------------
def save_preds(df, out_path):
    try:
        df.to_csv("preds.csv", index=False)
        debug(f"File 'preds.csv' saved locally. Uploading to: {out_path}")
        cmd = f"gsutil cp preds.csv {out_path}"
        result = os.system(cmd)
        if result != 0:
            debug(f"[ERROR] Failed to upload preds.csv to {out_path} (exit code {result})")
            sys.exit(1)
        debug("Predictions uploaded to GCS successfully.")
    except Exception as e:
        debug(f"[ERROR] Failed during save_preds: {e}")
        traceback.print_exc()
        sys.exit(1)

# ---------------- Run inference ----------------
def run_inference(model_uri, eval_csv):
    debug(f"Loading model: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
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
    debug(f"Inference results saved to {out_dir}")

    # ---------------- Upload best model to GCS ----------------
    local_model_dir = f"best_model_{timestamp}"
    debug(f"Saving MLflow model locally to: {local_model_dir}")
    try:
        mlflow.pyfunc.save_model(model, path=local_model_dir)
        gcs_model_path = f"{GCS_BUCKET}/models/best_model_{timestamp}"
        debug(f"Uploading best model to GCS: {gcs_model_path}")
        ret = os.system(f"gsutil -m cp -r {local_model_dir} {gcs_model_path}")
        if ret != 0:
            debug(f"[ERROR] Failed to upload model to GCS (exit code {ret})")
        else:
            debug("Best model successfully uploaded to GCS ✅")
        # Clean up local saved model folder
        shutil.rmtree(local_model_dir)
    except Exception as e:
        debug(f"[ERROR] Failed to save/upload best model: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"[SUCCESS] Inference completed and best model uploaded to {gcs_model_path}", flush=True)

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

    best_run_id = runs[0].info.run_id
    best_model_uri = f"runs:/{best_run_id}/model"
    print(f"Using best model from run {best_run_id}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, default=best_model_uri)
    parser.add_argument("--eval_csv", type=str, default=f"{GCS_BUCKET}/data/processed/eval.csv")
    args = parser.parse_args()

    try:
        run_inference(args.model_uri, args.eval_csv)
        debug("Inference script completed successfully ✅")
    except Exception as e:
        debug(f"[FATAL] Inference script failed: {e}")
        traceback.print_exc()
        sys.exit(1)
