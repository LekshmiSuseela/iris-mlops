import os, joblib, time, json, sys, traceback
from datetime import datetime
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- Configuration ----------------
GCS_BUCKET = "gs://mlops-474118-artifacts"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris_classification")

# ---------------- RandomForest hyperparameters ----------------
RF_PARAMS = {
    "n_estimators": 200,       # Number of trees
    "max_depth": 5,            # Max depth of each tree
    "min_samples_split": 4,    # Minimum samples to split a node
    "min_samples_leaf": 2,     # Minimum samples in a leaf
    "random_state": 42
}

# ---------------- Debug Utility ----------------
def debug(msg: str):
    """Standardized debug print with UTC timestamp."""
    print(f"[{datetime.utcnow().isoformat()}] [DEBUG] {msg}", flush=True)

# ---------------- Data Loading in batches ----------------
def load_data_in_batches(gcs_path, batch_size=20):
    debug(f"Loading dataset in batches of {batch_size} from: {gcs_path}")
    try:
        chunks = pd.read_csv(gcs_path, chunksize=batch_size)
        df = pd.concat(chunks, ignore_index=True)
        debug(f"Dataset loaded with shape {df.shape} and columns: {list(df.columns)}")
        return df
    except Exception as e:
        debug(f"[ERROR] Failed to load data in batches from {gcs_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

# ---------------- Training Logic ----------------
def train(data_path):
    debug(f"Starting training process with data path: {data_path}")
    df = load_data_in_batches(data_path, batch_size=20)

    # Identify target column
    try:
        if 'target' in df.columns:
            y = df['target']
            X = df.drop(columns=['target'])
            debug("Using 'target' as the label column.")
        elif 'species' in df.columns:
            y = df['species']
            X = df.drop(columns=['species'])
            debug("Using 'species' as the label column.")
        else:
            raise ValueError("Expected 'target' or 'species' column in dataset.")
    except Exception as e:
        debug(f"[ERROR] Target column not found: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Train-test split
    debug("Splitting data into training and evaluation sets...")
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    debug(f"Data split complete: Train={X_train.shape}, Eval={X_eval.shape}")

    # Model training
    debug(f"Initializing RandomForestClassifier with params: {RF_PARAMS}")
    clf = RandomForestClassifier(**RF_PARAMS)
    debug("Training model...")
    clf.fit(X_train, y_train)
    debug("Model training complete.")

    # Evaluation
    preds = clf.predict(X_eval)
    acc = accuracy_score(y_eval, preds)
    debug(f"Evaluation complete. Accuracy={acc:.4f}")

    return clf, acc

# ---------------- Main ----------------
if __name__ == '__main__':
    try:
        data_path = os.getenv('DATA_CSV', f"{GCS_BUCKET}/data/iris.csv")
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        run_output_dir = f"{GCS_BUCKET}/training_runs/{timestamp}"

        debug("Starting new MLflow run...")
        debug(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        debug(f"Run output directory: {run_output_dir}")

        with mlflow.start_run():
            clf, acc = train(data_path)

            # Log parameters and metrics
            debug("Logging parameters and metrics to MLflow...")
            for k, v in RF_PARAMS.items():
                mlflow.log_param(k, v)
            mlflow.log_metric('accuracy', float(acc))

            # Save model locally
            local_model = f"model_{timestamp}.pkl"
            joblib.dump(clf, local_model)
            debug(f"Model saved locally as {local_model}")

            # Log model to MLflow
            debug("Logging model to MLflow artifact store...")
            mlflow.sklearn.log_model(clf, "model")
            debug("Model successfully logged to MLflow.")

            # Copy model to GCS
            debug(f"Uploading model to GCS path: {run_output_dir}/model.pkl")
            ret = os.system(f"gsutil cp {local_model} {run_output_dir}/model.pkl")
            if ret == 0:
                debug("Model successfully uploaded to GCS.")
            else:
                debug(f"[ERROR] Failed to upload model to GCS (exit code {ret}).")

            debug(f"Run complete. Final model accuracy: {acc:.4f}")

    except Exception as e:
        debug(f"[FATAL] Training pipeline crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
