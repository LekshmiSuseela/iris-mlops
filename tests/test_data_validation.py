from scripts.data_validation import validate_data

def test_validate():
    try:
        validate_data('gs://mlops-474118-artifacts/data/iris.csv')
    except AssertionError as e:
        assert False, str(e)
