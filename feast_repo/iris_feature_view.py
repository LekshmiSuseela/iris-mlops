import logging
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32

# --------------------------
# Setup debug logging
# --------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.debug("Initializing Feast feature definitions for Iris dataset...")

try:
    # --------------------------
    # Entity Definition
    # --------------------------
    logging.debug("Creating entity: iris_id")
    iris_entity = Entity(
        name="iris_id",
        value_type=ValueType.INT64,
        description="Unique identifier for each Iris record."
    )
    logging.debug(f"Entity created successfully: {iris_entity}")

    # --------------------------
    # File Source Definition
    # --------------------------
    file_path = "gs://mlops-474118-artifacts/data/features.parquet"
    logging.debug(f"Creating FileSource for path: {file_path}")

    iris_source = FileSource(
        path=file_path,
        timestamp_field="event_timestamp"
    )
    logging.debug(f"FileSource created successfully: {iris_source}")

    # --------------------------
    # Feature View Definition
    # --------------------------
    logging.debug("Defining FeatureView: iris_features")

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

    logging.debug("FeatureView 'iris_features' defined successfully.")
    logging.info("All Feast feature definitions initialized successfully.")

except Exception as e:
    logging.exception("Error while defining Feast features:")
    raise
