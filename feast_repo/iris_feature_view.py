from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast import FileSource
from feast.data_format import FileFormat
#from feast.data_format import CsvFormat
import pandas as pd

# Define your entity (unique identifier)
iris_entity = Entity(name="iris_id", join_keys=["iris_id"])

# Read your CSV
df = pd.read_csv('gs://mlops-474118-artifacts/data/iris.csv')

# Convert to Parquet
df.to_parquet('gs://mlops-474118-artifacts/data/iris.parquet', index=False)

# Define your data source
iris_source = FileSource(
    path = "gs://mlops-474118-artifacts/data/iris.parquet",
    timestamp_field = "event_timestamp",  # Add this column if not present
)

# Define your Feature View
iris_fv = FeatureView(
    name = "iris_features",
    entities = [iris_entity],
    ttl = timedelta(days=1),
    schema = [
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    source = iris_source,
)
