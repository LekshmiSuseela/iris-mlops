import os
import sys
import traceback
import joblib
import argparse
from datetime import datetime
import pandas as pd
import mlflow.pyfunc

GCS_BUCKET = "gs://mlops-474118-artifacts"
MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

def debug(msg: str):
    """Unified debug logger."""
    print(f"[DEBUG] {msg}", flush=True)


def fetch_eval_data(path):
    debug(f"Fetching evaluation data from: {path}")
    try:
        # Handle GCS path: copy locally for pandas read
        if path.startswith("gs://"):
            local_path = "data/tmp_eval.csv"
            os.makedirs("data", exist_ok=True)
            cmd = f"gsutil cp {path} {local_path}"
            debug(f"Running command: {cmd}")
            result = os.system(cmd)
            if result != 0:
                debug(f"[ERROR] gsutil copy failed with exit code {result}")
                sys.exit(1)
            debug("File successfully copied from GCS.")
            path = local_path
        else:
            debug("Detected local file path.")

        df = pd.read_csv(path)
        debug(f"Successfully read eval CSV with shape: {df.shape}")
        debug(f"Columns found: {list(df.columns)}")
        return df
    except Exception as e:
        debug(f"[ERROR] Failed to fetch or read eval data: {e}")
        traceback.print_exc()
        sys.exit(1)


def save_preds(df, out_path):
    debug("Saving predictions to local file 'preds.csv'")
    try:
        df.to_csv("preds.csv", index=False)
        debug(f"File 'preds.csv' saved successfully. Uploading to: {out_path}")
        cmd = f"gsutil cp preds.csv {out_path}"
        debug(f"Running command: {cmd}")
        result = os.system(cmd)
        if result != 0:
            debug(f"[ERROR] Failed to upload preds.csv to {out_path} (exit code {result})")
            sys.exit(1)
        debug("Predictions uploaded to GCS successfully.")
    except Exception as e:
        debug(f"[ERROR] Failed during save_preds: {e}")
        traceback.print_exc()
        sys.exit(1)


def run_inference(model_uri, eval_csv):
    debug("Starting inference process...")
    debug(f"Model URI: {model_uri}")
    debug(f"Eval CSV: {eval_csv}")

    try:
        debug("Loading MLflow model...")
        model = mlflow.pyfunc.load_model(model_uri)
        debug("Model loaded successfully.")
    except Exception as e:
        debug(f"[ERROR] Failed to load model from MLflow URI: {e}")
        traceback.print_exc()
        sys.exit(1)

    df = fetch_eval_data(eval_csv)

    # Detect feature columns
    debug("Preparing features for prediction...")
    if 'target' in df.columns:
        debug("Dropping 'target' column.")
        X = df.drop(columns=['target'])
    elif 'species' in df.columns:
        debug("Dropping 'species' column.")
        X = df.drop(columns=['species'])
    else:
        debug("No target/species column found. Using all columns.")
        X = df

    # Run inference
    try:
        debug("Running model predictions...")
        preds = model.predict(X)
        debug(f"Predictions completed. Sample: {preds[:5]}")
    except Exception as e:
        debug(f"[ERROR] Model prediction failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Append predictions
    df['prediction'] = preds
    debug("Predictions column added to DataFrame.")

    # Save to GCS
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = f"{GCS_BUCKET}/inference/{timestamp}"
    debug(f"Saving output to GCS directory: {out_dir}")
    save_preds(df, f"{out_dir}/preds.csv")

    print(f"[SUCCESS] Inference completed and saved to {out_dir}", flush=True)


if __name__ == "__main__":
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri="http://<MLFLOW_SERVER_IP>:5000")

    # Replace with your actual experiment ID
    experiment_id = "1"

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"],  # Assuming you log accuracy
        max_results=1)
    
    best_run_id = runs[0].info.run_id
    best_model_uri = f"runs:/{best_run_id}/model"
    print(f"Using best model from run {best_run_id}")

    # Load model
    model = mlflow.pyfunc.load_model(best_model_uri)

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
