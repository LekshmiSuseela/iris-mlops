import pandas as pd

def validate_data(path):
    df = pd.read_csv(path)
    assert not df.isnull().values.any(), "Null values found in data"
    assert df.shape[0] > 0, "Empty dataset"
    print('Data validation passed for', path)

if __name__ == '__main__':
    validate_data('gs://mlops-474118-artifacts/data/iris.csv')
