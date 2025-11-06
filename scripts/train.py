import os, joblib, time, json
from datetime import datetime
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

GCS_BUCKET = "gs://mlops-474118-artifacts"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://<MLFLOW_SERVER_IP>:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris_classification")

def load_data(gcs_path):
    # gcs_path example: gs://mlops-474118-artifacts/data/raw/iris.csv
    return pd.read_csv(gcs_path)

def train(data_path):
    df = load_data(data_path)
    # expecting target column 'target' or 'species'
    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    elif 'species' in df.columns:
        y = df['species']
        X = df.drop(columns=['species'])
    else:
        raise ValueError("Expected 'target' or 'species' column in dataset.")

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_eval)
    acc = accuracy_score(y_eval, preds)
    return clf, acc

if __name__ == '__main__':
    data_path = os.getenv('DATA_CSV', f"{GCS_BUCKET}/data/raw/iris.csv")
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    run_output_dir = f"{GCS_BUCKET}/training_runs/{timestamp}"
    with mlflow.start_run():
        clf, acc = train(data_path)
        mlflow.log_param('model_type', 'RandomForest')
        mlflow.log_metric('accuracy', float(acc))
        # save model to local and then copy to GCS
        local_model = f"model_{timestamp}.pkl"
        joblib.dump(clf, local_model)
        mlflow.sklearn.log_model(clf, "model")
        # copy to GCS
        os.system(f"gsutil cp {local_model} {run_output_dir}/model.pkl")
        print('Run complete. Accuracy:', acc)
