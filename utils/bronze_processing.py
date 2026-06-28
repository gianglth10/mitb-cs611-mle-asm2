import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

source_file = {'features_attributes': 'data/features_attributes.csv',
               'features_financials': 'data/features_financials.csv',
               'lms_loan_daily': 'data/lms_loan_daily.csv',
               'feature_clickstream': 'data/feature_clickstream.csv'}

def check_bronze_table(bronze_table_directory, spark, source_name):
    """
    Check the bronze table for a given source_name and return available snapshot dates.
    """
    # Load source name & Check availability
    csv_file_path = source_file.get(source_name)

    if csv_file_path is None:
        print(f"Invalid source_name: {source_name}")
        print(f"Available source names: {list(source_file.keys())}")
        return None

    # Check if file_name exists in dictionary
    if csv_file_path is None:
        raise ValueError(
            f"Invalid source_name: {source_name}. "
            f"Available options are: {list(source_file.keys())}"
        )

    # Check if csv file path exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")

    # Load data
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Check table info.
    print(f"Loaded file: {csv_file_path}")
    print(f"File path: {csv_file_path}")
    print("Row count:", df.count())
    print("Column count:", len(df.columns))

    if "Customer_ID" in df.columns: 
        print("Unique customer count:")
        df.agg(F.countDistinct("Customer_ID").alias("unique_customer_count")).show()

    print("Schema:")
    df.printSchema()

    print("Table statistics:")
    display(df.describe().toPandas())

    # Get available snapshot dates
    if "snapshot_date" not in df.columns:
        raise ValueError(
            f"No 'snapshot_date' column found in source_name='{source_name}'. "
            f"Available columns are: {df.columns}"
        )

    available_dates = (df.select("snapshot_date").distinct().orderBy("snapshot_date").toPandas()["snapshot_date"].tolist())
    print("Available snapshot dates:")
    print([date.strftime("%Y-%m-%d") for date in available_dates])

    return available_dates

def process_bronze_table(snapshot_date_str, bronze_table_directory, spark, source_name=None):
    """
    Process the bronze table for a given snapshot_date and source_name, and save the filtered data to a CSV file.
    """    
    # Load source name & Check availability
    csv_file_path = source_file.get(source_name)

    if csv_file_path is None:
        print(f"Invalid source_name: {source_name}")
        print(f"Available source names: {list(source_file.keys())}")
        return None

    # Create file_path 
    bronze_directory = os.path.join(bronze_table_directory, source_name)
    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)

    if os.path.exists(filepath):
        print("Skipped existing bronze table:", filepath)
        return None

    # Load data
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Filter data by snapshot_date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    if 'snapshot_date' in df.columns:
        df = (df.withColumn("snapshot_date", F.to_date(col("snapshot_date"))).filter(col("snapshot_date") == F.lit(snapshot_date).cast(DateType())))
    else:
        raise ValueError(f"No 'snapshot_date' column in {source_name}")

    df = (df.withColumn("bronze_ingested_at", F.current_timestamp())
          .withColumn("bronze_source_file", F.lit(csv_file_path))
          .withColumn("bronze_source_name", F.lit(source_name))
          .withColumn("bronze_snapshot_date", F.lit(snapshot_date_str)))

    print(snapshot_date_str + " row count:", df.count())

    # Save to CSV
    df.toPandas().to_csv(filepath, index=False)
    print('Saved to:', filepath)

    return df
