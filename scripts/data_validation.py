import pandas as pd
import os
import sys
import traceback

def debug(msg: str):
    """Standard debug message printer."""
    print(f"[DEBUG] {msg}", flush=True)


def validate_data(path):
    debug("Starting data validation...")
    debug(f"Input path: {path}")
    debug(f"Current working directory: {os.getcwd()}")

    # Check if path is GCS or local
    if path.startswith("gs://"):
        debug("Detected GCS path. Copying file locally for validation...")
        local_path = "data/tmp_validation.csv"
        os.makedirs("data", exist_ok=True)

        # Try to copy from GCS to local
        cmd = f"gsutil cp {path} {local_path}"
        debug(f"Running command: {cmd}")
        result = os.system(cmd)
        if result != 0:
            debug(f"[ERROR] Failed to copy {path} from GCS. Exit code: {result}")
            sys.exit(1)
        path = local_path
        debug("File successfully copied from GCS.")
    else:
        debug("Detected local file path.")

    # Step 1: Verify file existence
    if not os.path.exists(path):
        debug(f"[ERROR] File not found at {path}")
        sys.exit(1)
    else:
        debug(f"File exists: {path}")

    # Step 2: Read CSV
    try:
        df = pd.read_csv(path)
        debug(f"Data read successfully. Shape: {df.shape}")
        debug(f"Columns: {list(df.columns)}")
    except Exception as e:
        debug(f"[ERROR] Failed to read CSV: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Validate for nulls
    if df.isnull().values.any():
        debug("[ERROR] Null values found in dataset.")
        null_summary = df.isnull().sum()
        debug(f"Null summary:\n{null_summary}")
        raise AssertionError("Null values found in data")
    else:
        debug("No null values found.")

    # Step 4: Validate non-empty dataset
    if df.shape[0] == 0:
        debug("[ERROR] Empty dataset detected.")
        raise AssertionError("Empty dataset")
    else:
        debug(f"Dataset has {df.shape[0]} rows — OK.")

    # Step 5: Success message
    print(f"[SUCCESS] Data validation passed for {path}")


if __name__ == '__main__':
    try:
        validate_data('gs://mlops-474118-artifacts/data/iris.csv')
        debug("Validation script completed successfully ✅")
    except Exception as e:
        debug(f"[FATAL] Validation failed: {e}")
