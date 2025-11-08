import pytest
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="session")
def load_data_and_model():
    data = pd.read_csv('gs://mlops-474118-artifacts/data/iris.csv')
    model = load("gs://mlops-474118-artifacts/models/best_model_20251108T121848Z/model/model.pkl")
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
