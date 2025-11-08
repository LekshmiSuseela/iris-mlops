import pandas as pd
import numpy as np
import os
from google.cloud import storage
import sys
import traceback

BUCKET = 'mlops-474118-artifacts'
RAW_PATH = 'data/iris.csv'
OUT_PATH = 'data/iris_v2.csv'


def debug(msg: str):
    """Print debug messages with a consistent format."""
    print(f"[DEBUG] {msg}", flush=True)


def augment():
    debug("Starting augmentation process...")
    debug(f"Current working directory: {os.getcwd()}")

    # Check Python environment info
    debug(f"Python executable: {sys.executable}")
    debug(f"Python version: {sys.version}")

    # Construct GCS URI
    if RAW_PATH.startswith("gs://"):
        uri = RAW_PATH
    else:
        uri = f"gs://{BUCKET}/{RAW_PATH}"
    debug(f"Resolved input URI: {uri}")

    # Step 1: Verify data file existence
    if not os.path.exists(RAW_PATH):
        debug(f"[ERROR] Local file {RAW_PATH} not found. Attempting to list directory:")
        os.system("ls -R data || true")
        raise FileNotFoundError(f"{RAW_PATH} not found locally!")

    debug("Found input CSV file. Reading into pandas...")

    try:
        df = pd.read_csv(RAW_PATH)
        debug(f"Data read successfully. Shape: {df.shape}")
        debug(f"Columns: {list(df.columns)}")
    except Exception as e:
        debug(f"[ERROR] Failed to read CSV: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Perform augmentation
    debug("Performing Gaussian noise augmentation on numeric columns...")
    df_aug = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    debug(f"Numeric columns found: {list(num_cols)}")

    for c in num_cols:
        df_aug[c] = df_aug[c] + np.random.normal(0, 0.01, size=df_aug.shape[0])
        debug(f"Added noise to column: {c}")

    # Step 3: Save augmented data
    debug(f"Saving augmented dataset to {OUT_PATH}...")
    try:
        df_aug.to_csv(OUT_PATH, index=False)
        debug("File saved locally.")
    except Exception as e:
        debug(f"[ERROR] Failed to save augmented CSV: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Upload to GCS
    upload_cmd = f"gsutil cp {OUT_PATH} gs://{BUCKET}/{OUT_PATH}"
    debug(f"Uploading file to GCS: {upload_cmd}")

    result = os.system(upload_cmd)
    if result != 0:
        debug(f"[ERROR] GCS upload failed with exit code {result}")
        sys.exit(1)

    debug("Upload completed successfully ✅")
    print(f"Augmented data uploaded to gs://{BUCKET}/data/iris_v2.csv")


if __name__ == "__main__":
    try:
        augment()
        debug("Augmentation script finished successfully ✅")
    except Exception as e:
        debug(f"[FATAL] Script failed: {e}")
        traceback.print_exc()
        sys.exit(1)
