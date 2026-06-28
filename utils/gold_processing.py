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

PII_COLUMNS = ["Name", "SSN"]

def drop_pii_columns(df, pii_columns=PII_COLUMNS):
    """
    Drop personally identifiable information (PII) columns before creating Gold feature store.
    """
    existing_pii_cols = [c for c in pii_columns if c in df.columns]

    if len(existing_pii_cols) > 0:
        df = df.drop(*existing_pii_cols)

    return df
    
def impute_gold_feature_store_nulls(feature_df, verbose=True):
    """
    Impute null values in the Gold feature store.

    Numeric columns:
    - Use median if distribution is highly skewed.
    - Use mean if distribution is not highly skewed.
    - Use 0 if the whole column is null.

    String columns:
    - Fill null values with "Unknown".

    Key columns such as loan_id and Customer_ID should not be imputed.
    Rows with missing keys are removed because they cannot be used reliably for ML training.
    """

    # Remove rows with missing key values
    feature_df = feature_df.filter(F.col("loan_id").isNotNull() & F.col("Customer_ID").isNotNull())

    # Identify numeric columns
    numeric_columns = [name for name, dtype in feature_df.dtypes if dtype in ["int", "bigint", "float", "double"]]

    # Identify string columns except key/version columns
    string_columns = [name for name, dtype in feature_df.dtypes if dtype == "string" and name not in ["loan_id", "Customer_ID", "feature_store_version"]]

    imputation_rules = []

    # Impute numeric columns using mean / median logic
    for column_name in numeric_columns:
        null_count = feature_df.filter(F.col(column_name).isNull()).count()

        if null_count == 0:
            continue
        
        stats = feature_df.select(
            F.count(F.col(column_name)).alias("non_null_count"),
            F.mean(F.col(column_name)).alias("mean"),
            F.expr(f"percentile_approx(`{column_name}`, 0.5)").alias("median"),
            F.skewness(F.col(column_name)).alias("skewness")).collect()[0]

        non_null_count = stats["non_null_count"]
        mean_value = stats["mean"]
        median_value = stats["median"]
        skewness = stats["skewness"]

        if non_null_count == 0:
            impute_value = 0
            method = "zero"
        elif skewness is not None and abs(skewness) >= 1:
            impute_value = median_value
            method = "median"
        else:
            impute_value = mean_value
            method = "mean"

        if impute_value is None:
            impute_value = 0

        feature_df = feature_df.withColumn(column_name, F.coalesce(F.col(column_name), F.lit(impute_value)))
        imputation_rules.append({
            "column_name": column_name,
            "method": method,
            "value": impute_value,
            "skewness": skewness})

    # Impute string / categorical columns
    for column_name in string_columns:
        null_count = feature_df.filter(F.col(column_name).isNull()).count()

        if null_count == 0:
            continue
            
        feature_df = feature_df.withColumn(column_name, F.coalesce(F.col(column_name), F.lit("Unknown")))
        imputation_rules.append({
            "column_name": column_name,
            "method": "Unknown",
            "value": "Unknown",
            "skewness": None})

    if verbose:
        print("Gold imputation rules applied.")
        # print(pd.DataFrame(imputation_rules).to_string(index=False))

    return feature_df

def process_labels_gold_table(snapshot_date_str, silver_table_directory, gold_label_store_directory, spark, dpd, mob, verbose=True):
    """
    Build Gold-level label store.
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(silver_table_directory, "lms_loan_daily", partition_name)
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))
    df = df.withColumn("label_snapshot_date", F.col("snapshot_date"))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date", "label_snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    os.makedirs(gold_label_store_directory, exist_ok=True)
    filepath = os.path.join(gold_label_store_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, silver_table_directory, gold_feature_store_directory, spark, mob, dpd=30, verbose=True):
    """
    Build Gold-level feature store.

    Main steps:
    1. Use customer features from feature_snapshot_date.
    2. Use target MOB loan records from feature_snapshot_date + mob as the base population.
    3. Avoid repayment/outcome fields as model features because those are not known at application time.
    4. Impute remaining null values using mean / median / Unknown strategy.
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    snapshot_date_key = snapshot_date_str.replace("-", "_")
    label_snapshot_date = snapshot_date + relativedelta(months=mob)
    label_snapshot_date_str = label_snapshot_date.strftime("%Y-%m-%d")
    label_snapshot_date_key = label_snapshot_date_str.replace("-", "_")

    # Load source name & Check availability
    source_file = {
        "features_attributes": f"{silver_table_directory}/features_attributes/silver_features_attributes_{snapshot_date_key}.parquet",
        "features_financials": f"{silver_table_directory}/features_financials/silver_features_financials_{snapshot_date_key}.parquet",
        "lms_loan_daily": f"{silver_table_directory}/lms_loan_daily/silver_lms_loan_daily_{label_snapshot_date_key}.parquet",
        "feature_clickstream": f"{silver_table_directory}/feature_clickstream/silver_feature_clickstream_{snapshot_date_key}.parquet",
    }
    
    attributes_df = spark.read.parquet(source_file["features_attributes"])
    financials_df = spark.read.parquet(source_file["features_financials"])
    loan_df = spark.read.parquet(source_file["lms_loan_daily"])
    clickstream_df = spark.read.parquet(source_file["feature_clickstream"])

    if verbose:
        print("Loaded loan:", loan_df.count())
        print("Loaded attributes:", attributes_df.count())
        print("Loaded financials:", financials_df.count())
        print("Loaded clickstream:", clickstream_df.count())

    # Base population: loans that mature to target MOB at the future label snapshot.
    # Only identifiers and snapshot dates are retained here; repayment fields are labels/outcomes.
    base_loans_df = (loan_df
                     .filter(F.col("mob") == mob)
                     .select("loan_id", "Customer_ID", "dpd")
                     .dropDuplicates(["loan_id", "Customer_ID"])
                     .withColumn("label", F.when(F.col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
                     .withColumn("label_def", F.lit(str(dpd) + "dpd_" + str(mob) + "mob").cast(StringType()))
                     .drop("dpd")
                     .withColumn("feature_snapshot_date", F.lit(snapshot_date_str).cast(DateType()))
                     .withColumn("label_snapshot_date", F.lit(label_snapshot_date_str).cast(DateType())))

    # Drop PII columns before selecting customer-level attributes
    attributes_df = drop_pii_columns(attributes_df)
    
    # Customer-level attributes
    attributes_features_df = (attributes_df.select("Customer_ID", "Age", "Occupation").dropDuplicates(["Customer_ID"]))

    # Customer financial features
    financials_features_df = (financials_df.select(
            "Customer_ID",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Credit_Mix_Score",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Number_Credit_History_Age_Month",
            "Payment_of_Min_Amount_Score",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
            "Loan_Type_Count",
            "Payment_Spending_Score",
            "Payment_Value_Score"
        ).dropDuplicates(["Customer_ID"]))

    # Derived financial ratios available at feature snapshot time.
    financials_features_df = (
        financials_features_df
        .withColumn(
            "debt_to_income_ratio",
            F.when((F.col("Annual_Income").isNotNull()) & (F.col("Annual_Income") > 0),
                   F.col("Outstanding_Debt") / F.col("Annual_Income")).otherwise(None))
        .withColumn(
            "emi_to_salary_ratio",
            F.when((F.col("Monthly_Inhand_Salary").isNotNull()) & (F.col("Monthly_Inhand_Salary") > 0),
                   F.col("Total_EMI_per_month") / F.col("Monthly_Inhand_Salary")).otherwise(None))
        .withColumn(
            "investment_to_salary_ratio",
            F.when((F.col("Monthly_Inhand_Salary").isNotNull()) & (F.col("Monthly_Inhand_Salary") > 0),
                   F.col("Amount_invested_monthly") / F.col("Monthly_Inhand_Salary")).otherwise(None))
    )

    # Clickstream features
    fe_columns = [c for c in clickstream_df.columns if c.startswith("fe_")]
    clickstream_features_df = (clickstream_df.select("Customer_ID", *fe_columns).dropDuplicates(["Customer_ID"]))

    # Join all feature groups
    feature_df = (base_loans_df
                  .join(attributes_features_df, on="Customer_ID", how="left")
                  .join(financials_features_df, on="Customer_ID", how="left")
                  .join(clickstream_features_df, on="Customer_ID", how="left"))

    # Remove records with missing keys
    feature_df = feature_df.filter(F.col("loan_id").isNotNull() & F.col("Customer_ID").isNotNull())

    # Impute remaining null values after joining all feature groups
    feature_df = impute_gold_feature_store_nulls(feature_df, verbose=verbose)

    # Final null check
    # Final null check
    feature_row_count = feature_df.count()
    if feature_row_count == 0:
        print(f"Skipped {snapshot_date_str}: no feature rows generated for mob = {mob}")
        return feature_df
    
    null_counts = feature_df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in feature_df.columns]).collect()[0].asDict()
    remaining_nulls = {column_name: null_count for column_name, null_count in null_counts.items() if (null_count or 0) > 0}

    if len(remaining_nulls) > 0:
        raise ValueError(f"Gold feature store still has null values: {remaining_nulls}")

    # Save Gold feature store
    os.makedirs(gold_feature_store_directory, exist_ok=True)

    output_path = os.path.join(
        gold_feature_store_directory,
        f"gold_feature_store_{snapshot_date_key}.parquet"
    )

    feature_df.write.mode("overwrite").parquet(output_path)

    print("Saved feature store to:", output_path)
    print("Feature store row count:", feature_df.count())

    return feature_df
