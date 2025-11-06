import os, joblib, argparse
from datetime import datetime
import pandas as pd
import mlflow.pyfunc

GCS_BUCKET = "gs://mlops-474118-artifacts"

def fetch_eval_data(path):
    return pd.read_csv(path)

def save_preds(df, out_path):
    df.to_csv('preds.csv', index=False)
    os.system(f"gsutil cp preds.csv {out_path}")

def run_inference(model_uri, eval_csv):
    model = mlflow.pyfunc.load_model(model_uri)
    df = fetch_eval_data(eval_csv)
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
    elif 'species' in df.columns:
        X = df.drop(columns=['species'])
    else:
        X = df
    preds = model.predict(X)
    df['prediction'] = preds
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = f"{GCS_BUCKET}/inference/{timestamp}"
    save_preds(df, f"{out_dir}/preds.csv")
    print('Inference saved to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uri', type=str, default="models:/iris_best_model/Production")
    parser.add_argument('--eval_csv', type=str, default=f"{GCS_BUCKET}/data/processed/eval.csv")
    args = parser.parse_args()
    run_inference(args.model_uri, args.eval_csv)
