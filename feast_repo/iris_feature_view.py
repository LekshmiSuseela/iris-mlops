# Simple example feature view (placeholder).
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32

iris_entity = Entity(name="iris_id", value_type=ValueType.INT64, description="iris id")

# This is a placeholder FileSource. In production use BigQuery or another supported source.
iris_source = FileSource(
    path="gs://mlops-474118-artifacts/data/features.parquet",
    timestamp_field="event_timestamp"
)

iris_fv = FeatureView(
    name="iris_features",
    entities=["iris_id"],
    ttl=None,
    features=[
        Feature(name="sepal_length", dtype=Float32),
        Feature(name="sepal_width", dtype=Float32),
        Feature(name="petal_length", dtype=Float32),
        Feature(name="petal_width", dtype=Float32),
    ],
    online=False,
    input=iris_source
)
