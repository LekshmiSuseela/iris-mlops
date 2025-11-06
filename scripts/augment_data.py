import pandas as pd, numpy as np, os
from google.cloud import storage

BUCKET = 'mlops-474118-artifacts'
RAW_PATH = 'data/raw/iris.csv'
OUT_PATH = 'data/raw/iris_v2.csv'

def augment():
    # read from GCS
    uri = f"gs://{BUCKET}/{RAW_PATH[5:]}" if RAW_PATH.startswith('gs://') else RAW_PATH
    # For simplicity, assume file already synced to Workbench. This script demonstrates augmentation steps.
    df = pd.read_csv('data/raw/iris.csv')
    df_aug = df.copy()
    # add gaussian noise to numeric cols
    num_cols = df.select_dtypes(include='number').columns
    for c in num_cols:
        df_aug[c] = df_aug[c] + np.random.normal(0, 0.01, size=df_aug.shape[0])
    df_aug.to_csv('data/raw/iris_v2.csv', index=False)
    os.system(f"gsutil cp data/raw/iris_v2.csv gs://{BUCKET}/data/raw/iris_v2.csv")
    print('Augmented data uploaded to GCS')

if __name__ == '__main__':
    augment()
