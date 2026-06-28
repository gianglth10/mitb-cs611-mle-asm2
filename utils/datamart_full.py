import os
import shutil
from datetime import datetime
from pathlib import Path

import pyspark
import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta

from utils.bronze_processing import process_bronze_table, source_file
from utils.gold_processing import process_features_gold_table, process_labels_gold_table
from utils.silver_processing import process_silver_table_all


DATAMART_DIR = Path("datamart")
BRONZE_DIR = DATAMART_DIR / "bronze"
SILVER_DIR = DATAMART_DIR / "silver"
GOLD_DIR = DATAMART_DIR / "gold"
GOLD_LABEL_STORE_DIR = GOLD_DIR / "label_store"
GOLD_FEATURE_STORE_DIR = GOLD_DIR / "feature_store"
GOLD_LABEL_STAGE_DIR = GOLD_DIR / "_label_store_by_snapshot"
GOLD_FEATURE_STAGE_DIR = GOLD_DIR / "_feature_store_by_snapshot"

DPD_CUTOFF = 30
MOB_CUTOFF = 6

SOURCE_NAMES = [
    "features_attributes",
    "features_financials",
    "lms_loan_daily",
    "feature_clickstream",
]

FEATURE_SOURCE_NAMES = [
    "features_attributes",
    "features_financials",
    "feature_clickstream",
]

STRING_COLUMNS = {
    "Customer_ID",
    "loan_id",
    "label_def",
    "Occupation",
    "feature_store_version",
}

DATE_COLUMNS = {
    "snapshot_date",
    "feature_snapshot_date",
    "label_snapshot_date",
}


def _spark_session(app_name: str):
    """ 
    Create a SparkSession with the specified application name and configuration settings. 
    """
    spark = (
        pyspark.sql.SparkSession.builder
        .appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS", "8"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _reset_dir(path: Path) -> None:
    """
    Reset a directory by deleting it if it exists and then creating it.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def init_datamart() -> None:
    """ 
    Initialize the datamart directory structure by creating necessary directories for bronze, silver, and gold layers.
    """
    for layer in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
        layer.mkdir(parents=True, exist_ok=True)


def _snapshot_dates_for_source(spark, source_name: str) -> list[str]:
    """ 
    Get all unique snapshot dates for a given source.
    """
    csv_file_path = source_file[source_name]
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    if "snapshot_date" not in df.columns:
        raise ValueError(f"No snapshot_date column found in {csv_file_path}")
    rows = (
        df.select(F.to_date("snapshot_date").alias("snapshot_date"))
        .where(F.col("snapshot_date").isNotNull())
        .distinct()
        .orderBy("snapshot_date")
        .collect()
    )
    return [row["snapshot_date"].isoformat() for row in rows]


def _all_source_snapshot_dates(spark) -> dict[str, list[str]]:
    """ 
    Get all unique snapshot dates for all sources.
    """
    return {source_name: _snapshot_dates_for_source(spark, source_name) for source_name in SOURCE_NAMES}


def _parquet_children(path: Path) -> list[str]:
    """ 
    Get the paths of all parquet directories under the specified path.
    """
    if not path.exists():
        return []
    return [str(child) for child in sorted(path.iterdir()) if child.is_dir() and child.suffix == ".parquet"]


def _normalise_schema(df):
    """ 
    Normalise the schema of a DataFrame by casting columns to their appropriate types.
    """
    for column_name in df.columns:
        if column_name in DATE_COLUMNS:
            df = df.withColumn(column_name, F.col(column_name).cast("date"))
        elif column_name == "label":
            df = df.withColumn(column_name, F.col(column_name).cast("int"))
        elif column_name in STRING_COLUMNS:
            df = df.withColumn(column_name, F.col(column_name).cast("string"))
        else:
            df = df.withColumn(column_name, F.col(column_name).cast("double"))
    return df


def _write_final_table(spark, input_paths: list[str], output_path: Path, table_name: str) -> None:
    """
    Write the final table to the specified output path.
    """
    if not input_paths:
        raise RuntimeError(f"No staged parquet files found for {table_name}")
    _reset_dir(output_path)
    frames = [_normalise_schema(spark.read.parquet(path)) for path in input_paths]
    df = frames[0]
    for frame in frames[1:]:
        df = df.unionByName(frame, allowMissingColumns=True)
    df.coalesce(1).write.mode("overwrite").parquet(str(output_path))
    print(f"Saved {table_name}: {df.count():,} rows at {output_path}")


def build_bronze() -> None:
    """
    Build the bronze layer of the datamart. 
    """
    init_datamart()
    _reset_dir(BRONZE_DIR)
    spark = _spark_session("assignment_2_build_bronze")
    try:
        source_dates = _all_source_snapshot_dates(spark)
        for source_name, snapshot_dates in source_dates.items():
            for snapshot_date in snapshot_dates:
                process_bronze_table(
                    snapshot_date_str=snapshot_date,
                    bronze_table_directory=str(BRONZE_DIR),
                    spark=spark,
                    source_name=source_name,
                )
    finally:
        spark.stop()


def build_silver() -> None:
    """
    Build the silver layer of the datamart. 
    """
    init_datamart()
    _reset_dir(SILVER_DIR)
    spark = _spark_session("assignment_2_build_silver")
    try:
        source_dates = _all_source_snapshot_dates(spark)
        for source_name, snapshot_dates in source_dates.items():
            for snapshot_date in snapshot_dates:
                process_silver_table_all(
                    snapshot_date_str=snapshot_date,
                    bronze_table_directory=str(BRONZE_DIR),
                    silver_table_directory=str(SILVER_DIR),
                    spark=spark,
                    source_name=source_name,
                    verbose=False,
                )
    finally:
        spark.stop()


def build_gold_label_and_feature_store(dpd_cutoff: int = DPD_CUTOFF, mob_cutoff: int = MOB_CUTOFF) -> None:
    """
    Build the gold layer of the datamart, including both label and feature stores.
    """
    init_datamart()
    _reset_dir(GOLD_LABEL_STAGE_DIR)
    _reset_dir(GOLD_FEATURE_STAGE_DIR)
    spark = _spark_session("assignment_2_build_gold")
    try:
        source_dates = _all_source_snapshot_dates(spark)
        loan_dates = sorted(source_dates["lms_loan_daily"])
        feature_date_sets = [set(source_dates[source_name]) for source_name in FEATURE_SOURCE_NAMES]
        candidate_feature_dates = sorted(set.intersection(*feature_date_sets))

        for label_date in loan_dates:
            process_labels_gold_table(
                snapshot_date_str=label_date,
                silver_table_directory=str(SILVER_DIR),
                gold_label_store_directory=str(GOLD_LABEL_STAGE_DIR),
                spark=spark,
                dpd=dpd_cutoff,
                mob=mob_cutoff,
                verbose=False,
            )

        valid_feature_dates = []
        loan_date_set = set(loan_dates)
        for feature_date in candidate_feature_dates:
            label_date = (
                datetime.strptime(feature_date, "%Y-%m-%d").date()
                + relativedelta(months=mob_cutoff)
            ).isoformat()
            if label_date in loan_date_set:
                valid_feature_dates.append(feature_date)

        for feature_date in valid_feature_dates:
            process_features_gold_table(
                snapshot_date_str=feature_date,
                silver_table_directory=str(SILVER_DIR),
                gold_feature_store_directory=str(GOLD_FEATURE_STAGE_DIR),
                spark=spark,
                mob=mob_cutoff,
                dpd=dpd_cutoff,
                verbose=False,
            )

        _write_final_table(
            spark,
            _parquet_children(GOLD_LABEL_STAGE_DIR),
            GOLD_LABEL_STORE_DIR,
            "gold/label_store",
        )
        _write_final_table(
            spark,
            _parquet_children(GOLD_FEATURE_STAGE_DIR),
            GOLD_FEATURE_STORE_DIR,
            "gold/feature_store",
        )
    finally:
        spark.stop()


def build_gold() -> None:
    build_gold_label_and_feature_store()


def build_datamart() -> None:
    build_bronze()
    build_silver()
    build_gold_label_and_feature_store()
