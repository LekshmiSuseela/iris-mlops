import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from google.cloud import storage
import os

MODEL_BUCKET = "mlops-474118-artifacts"
MODEL_BLOB = "models/best_model_20251108T121848Z/model/model.pkl"
LOCAL_MODEL_PATH = "downloaded_model.pkl"

@pytest.fixture(scope="session")
def load_data_and_model():
    # Load dataset
    data = pd.read_csv('gs://mlops-474118-artifacts/data/iris.csv')
    
    # Download model from GCS to local path
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_BLOB)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    
    # Load the model
    model = joblib.load(LOCAL_MODEL_PATH)
    
    # Split data
    X = data.drop(columns=["species"])
    y = data["species"]
    
    return model, X, y

def test_model_loads(load_data_and_model):
    model, X, y = load_data_and_model
    assert model is not None, "Model failed to load"

def test_model_prediction_shape(load_data_and_model):
    model, X, y = load_data_and_model
    preds = model.predict(X)
    assert len(preds) == len(X), "Predictions and data size mismatch"

def test_model_accuracy(load_data_and_model):
    model, X, y = load_data_and_model
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.7, f"Accuracy too low: {acc:.2f}"
