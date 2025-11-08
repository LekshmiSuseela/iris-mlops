import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_loaded_and_eval():
    # This test expects a Production model registered under name 'iris_best_model'
    model_uri = "gs://mlops-474118-artifacts/models/best_model_20251108T121848Z/model/"
    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.read_csv('gs://mlops-474118-artifacts/data/processed/eval.csv')
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
        y = df['target']
    elif 'species' in df.columns:
        X = df.drop(columns=['species'])
        y = df['species']
    else:
        X = df; y = None
    preds = model.predict(X)
    if y is not None:
        acc = accuracy_score(y, preds)
        assert acc > 0.5, f"Low accuracy: {acc}"
