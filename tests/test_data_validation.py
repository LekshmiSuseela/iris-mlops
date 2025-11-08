import pandas as pd

def test_data_not_empty():
    data = pd.read_csv('gs://mlops-474118-artifacts/data/iris.csv')
    assert not data.empty, "iris.csv is empty!"

def test_no_missing_values():
    data = pd.read_csv('gs://mlops-474118-artifacts/data/iris.csv')
    assert data.isnull().sum().sum() == 0, "There are missing values in iris.csv"

def test_expected_columns():
    data = pd.read_csv('gs://mlops-474118-artifacts/data/iris.csv')
    expected_cols = {'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species'}
    assert expected_cols.issubset(set(data.columns)), f"Columns missing! Found: {data.columns}"
