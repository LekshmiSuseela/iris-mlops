import os, joblib, sys, traceback
from datetime import datetime
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- Config ----------------
GCS_BUCKET = os.getenv("GCS_BUCKET", "gs://mlops-474118-artifacts")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris_classification")

BATCH_SIZE = 20  # For reading CSV in batches

# Hyperparameter grid
N_ESTIMATORS_LIST = [5, 100, 150]
MAX_DEPTH_LIST = [3, 5, 7]

# ---------------- Debug Utility ----------------
def debug(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] [DEBUG] {msg}", flush=True)

# ---------------- Load data in batches ----------------
def load_data_in_batches(path, batch_size=BATCH_SIZE):
    debug(f"Loading dataset in batches of {batch_size} from {path}")
    try:
        chunks = pd.read_csv(path, chunksize=batch_size)
        df = pd.concat(chunks, ignore_index=True)
        debug(f"Loaded dataset with shape {df.shape} and columns: {list(df.columns)}")
        return df
    except Exception as e:
        debug(f"[ERROR] Failed to load data in batches: {e}")
        traceback.print_exc()
        sys.exit(1)

# ---------------- Train & Evaluate ----------------
def train_and_evaluate(df, n_estimators, max_depth):
    # Identify target column
    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    elif 'species' in df.columns:
        y = df['species']
        X = df.drop(columns=['species'])
    else:
        raise ValueError("Dataset must have 'target' or 'species' column.")

    # Train-test split
    split_idx = int(0.8 * len(df))
    X_train, X_eval = X[:split_idx], X[split_idx:]
    y_train, y_eval = y[:split_idx], y[split_idx:]

    debug(f"Training RandomForest(n_estimators={n_estimators}, max_depth={max_depth})")
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_eval)
    acc = accuracy_score(y_eval, preds)
    debug(f"Evaluation complete. Accuracy={acc:.4f}")
    return clf, acc

# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        data_path = os.getenv('DATA_CSV', f"{GCS_BUCKET}/data/iris.csv")
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        run_output_dir = f"{GCS_BUCKET}/training_runs/{timestamp}"

        # Load data once in batches
        df = load_data_in_batches(data_path, batch_size=BATCH_SIZE)

        # Hyperparameter grid search
        for n_estimators in N_ESTIMATORS_LIST:
            for max_depth in MAX_DEPTH_LIST:
                with mlflow.start_run():
                    debug(f"Starting MLflow run for n_estimators={n_estimators}, max_depth={max_depth}")
                    clf, acc = train_and_evaluate(df, n_estimators, max_depth)

                    # Log params and metrics
                    mlflow.log_param("n_estimators", n_estimators)
                    mlflow.log_param("max_depth", max_depth)
                    mlflow.log_metric("accuracy", float(acc))

                    # Save model locally
                    local_model_path = f"model_{n_estimators}_{max_depth}_{timestamp}.pkl"
                    joblib.dump(clf, local_model_path)
                    debug(f"Model saved locally as {local_model_path}")

                    # Log model to MLflow
                    mlflow.sklearn.log_model(clf, artifact_path="model")
                    debug("Model logged to MLflow artifact store.")

                    # Optional: upload to GCS
                    gcs_model_path = f"{run_output_dir}/model_{n_estimators}_{max_depth}.pkl"
                    ret = os.system(f"gsutil cp {local_model_path} {gcs_model_path}")
                    if ret == 0:
                        debug(f"Model uploaded to GCS: {gcs_model_path}")
                    else:
                        debug(f"[ERROR] Failed to upload model to GCS (exit code {ret}).")

                    debug(f"Run complete. Accuracy={acc:.4f}")

        debug("All hyperparameter experiments completed ✅")
        # After all runs, write best accuracy to metrics.txt for CI
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("iris_classification")
            runs = client.search_runs([experiment.experiment_id], order_by=["metrics.accuracy DESC"], max_results=1)
            best_acc = runs[0].data.metrics["accuracy"]
            with open("metrics.txt", "w") as f:
                f.write(f"{best_acc:.4f}")
            debug(f"Best model accuracy written to metrics.txt: {best_acc:.4f}")
        except Exception as e:
            debug(f"[WARN] Could not extract accuracy for CI report: {e}")

    except Exception as e:
        debug(f"[FATAL] Training pipeline crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
