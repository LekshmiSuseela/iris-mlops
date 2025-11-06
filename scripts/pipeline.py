import argparse, subprocess, os
from datetime import datetime

def run_pipeline(data_version='v1'):
    print('Starting pipeline for data_version=', data_version)
    # In a Vertex AI pipeline you'd create components; here we run the scripts sequentially
    env = os.environ.copy()
    if data_version == 'v2':
        env['DATA_CSV'] = "gs://mlops-474118-artifacts/data/raw/iris_v2.csv"
    else:
        env['DATA_CSV'] = "gs://mlops-474118-artifacts/data/raw/iris.csv"
    env['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', 'http://<MLFLOW_SERVER_IP>:5000')
    subprocess.check_call(['python', 'scripts/train.py'], env=env)
    subprocess.check_call(['python', 'scripts/inference.py'], env=env)
    print('Pipeline finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', type=str, default='v1')
    args = parser.parse_args()
    run_pipeline(args.data_version)
