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

# ================================= LOAD DATA =================================
def load_bronze_table(snapshot_date_str, bronze_table_directory, spark, source_name, verbose=True):
    """
    Load the bronze table for a given snapshot_date and source_name, and return a Spark DataFrame.
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    snapshot_date_key = snapshot_date_str.replace("-", "_")

    # Load source name & Check availability
    source_file = {
        'features_attributes': f'{bronze_table_directory}/features_attributes/bronze_features_attributes_{snapshot_date_key}.csv',
        'features_financials': f'{bronze_table_directory}/features_financials/bronze_features_financials_{snapshot_date_key}.csv',
        'lms_loan_daily': f'{bronze_table_directory}/lms_loan_daily/bronze_lms_loan_daily_{snapshot_date_key}.csv',
        'feature_clickstream': f'{bronze_table_directory}/feature_clickstream/bronze_feature_clickstream_{snapshot_date_key}.csv',
    }
    
    csv_file_path = source_file.get(source_name)

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

    print('file name:', source_name)
    print('loaded from:', csv_file_path)
    return df

# ================================= Data transformation =================================
INVALID_STRINGS = ["", "na", "n/a", "nan", "none", "null", "_", "_______", "!@9#%8"]

def clean_string_value(df, column_name, verbose=True):
    """
    Clean string columns by trimming whitespace and converting invalid placeholders to null.

    Invalid values include null, empty strings, and predefined invalid strings such as "", "na", "n/a", "nan", "none", "null", "_", "_______", "nm".
    """
    if column_name not in df.columns:
        return df

    raw_col = F.trim(F.col(column_name).cast("string"))

    invalid_condition = (raw_col.isNull() | F.lower(raw_col).isin(INVALID_STRINGS))
    invalid_count = df.filter(invalid_condition).count()

    if verbose:
        if invalid_count > 0:
            invalid_values = (df.filter(invalid_condition).groupBy(column_name).count().orderBy(F.desc("count")).limit(20).collect())
            invalid_values_list = [row[column_name] for row in invalid_values]
            print(f"[{column_name} column] First 20 invalid string values: {invalid_values_list}")
            print(f">> {column_name} has {invalid_count} invalid string values >> Converted to null")
        elif invalid_count == 0:
            print(f"[{column_name} column] This column doesn't have invalid string values")
        print("-----")
   
    df = df.withColumn(column_name, F.when(invalid_condition, F.lit(None)).otherwise(raw_col))

    return df

def convert_to_numeric_value(df, column_name, target_type=FloatType(), verbose=True):
    """
    Convert messy numeric columns into valid numeric type.

    Cleaning logic:
    1. Invalid values: Values that are null, empty, or listed in INVALID_STRINGS
        e.g. "", "na", "n/a", "nan", "none", "null", "_", "_______", "!@9#%8"
        >> These values will be converted to null.

    2. Messy numeric values: Values that contain removable non-numeric characters but still represent valid numbers
         e.g. "1,000", "$2500", "3_500", " 4500 "
       >> These values will be cleaned by removing commas, underscores, currency symbols, spaces, and other invalid characters, then converted to the target numeric type.

    Note:
    - This function keeps digits, decimal points, and negative signs.
    - Columns with values that must be >= 0 should be handled separately using a range-checking function after numeric conversion.
    """
    if column_name not in df.columns:
        return df

    raw_col = F.trim(F.col(column_name).cast("string"))
    cleaned_col = F.regexp_replace(raw_col, r"[^0-9\.\-]", "")
    
    invalid_condition = (raw_col.isNull() | F.lower(raw_col).isin(INVALID_STRINGS))
    messy_condition = ((~invalid_condition) & (raw_col != cleaned_col))
    
    invalid_count = df.filter(invalid_condition).count()
    messy_count = df.filter(messy_condition).count()

    if verbose:
        if invalid_count > 0:
            invalid_values = (df.filter(invalid_condition).groupBy(column_name).count().orderBy(F.desc("count")).limit(20).collect())
            invalid_values_list = [row[column_name] for row in invalid_values]
    
            print(f"[{column_name} column] First 20 null/unusable values: {invalid_values_list}")
            print(f">> {column_name} has {invalid_count} null/unusable values >> Transformed to null")
        elif invalid_count == 0:
            print(f"[{column_name} column] This column doesn't have null/unusable numeric values")
        print("-----")

    if verbose:
        if messy_count > 0:
            messy_values = (df.filter(messy_condition).groupBy(column_name).count().orderBy(F.desc("count")).limit(20).collect())
            messy_values_list = [row[column_name] for row in messy_values]
    
            print(f"[{column_name} column] First 20 messy numeric-format values: {messy_values_list}")
            print(f">> {column_name} has {messy_count} messy numeric-format values >> Cleaned and converted to numeric values.")
        elif messy_count == 0:
            print(f"[{column_name} column] This column doesn't have messy numeric-format values")
        print("-----")

    df = df.withColumn(column_name, F.when(invalid_condition, F.lit(None).cast(target_type)).otherwise(cleaned_col.cast(target_type)))

    return df

def convert_credit_history_age_to_months(df, column_name="Credit_History_Age", verbose=True):
    """
    Convert 'X Years and Y Months' string to number of month
    """
    if column_name not in df.columns:
        return df

    years = F.regexp_extract(F.col(column_name).cast("string"), r"(\d+)\s+Years?", 1).cast("int")
    months = F.regexp_extract(F.col(column_name).cast("string"), r"(\d+)\s+Months?", 1).cast("int")

    df = df.withColumn('Number_Credit_History_Age_Month', F.when(F.col(column_name).isNull(), F.lit(None)).otherwise(F.coalesce(years, F.lit(0)) * 12 + F.coalesce(months, F.lit(0))))
    if verbose:
        print('Created "Number_Credit_History_Age_Month" column to reflect the number of credit history months.')
    
    return df

def filter_value_range(df, column_name, min_value=None, max_value=None, verbose=True):
    """
    Filter value based on range
    """
    if column_name not in df.columns:
        return df

    out_of_range_condition = F.lit(False)

    if min_value is not None:
        out_of_range_condition = out_of_range_condition | (F.col(column_name) < F.lit(min_value))

    if max_value is not None:
        out_of_range_condition = out_of_range_condition | (F.col(column_name) > F.lit(max_value))

    out_of_range_condition = F.col(column_name).isNotNull() & out_of_range_condition
    out_of_range_count = df.filter(out_of_range_condition).count()

    if min_value is not None and max_value is not None:
        range_text = f"valid range: {min_value} to {max_value}"
    elif min_value is not None:
        range_text = f"valid range: from {min_value}"
    elif max_value is not None:
        range_text = f"valid range: up to {max_value}"
    else:
        range_text = "no range specified"
    
    if verbose:
        if out_of_range_count > 0:
            invalid_values = (df.filter(out_of_range_condition).groupBy(column_name).count().orderBy(F.desc("count")).collect())
            invalid_values_list = [row[column_name] for row in invalid_values]
    
            print(f"[{column_name} column] First 20 invalid values: {invalid_values_list[:20]}")
            print(f">> {column_name} has {out_of_range_count} out-of-range values ({range_text}) >> Converted to null")
        elif out_of_range_count == 0:
            print(f"[{column_name} column] This column doesn't have out-of-range values")
        print("-----")

    df = df.withColumn(column_name, F.when(out_of_range_condition, F.lit(None)).otherwise(F.col(column_name)))

    return df

def standardize_ssn_value(df, column_name="SSN", verbose=True):
    """
    Clean and standardize SSN column.
    Valid format: 123-45-6789 or 123456789, which will be converted to 123-45-6789
    """
    if column_name not in df.columns:
        return df

    raw_col = F.trim(F.col(column_name).cast("string"))

    digits_values = F.regexp_replace(raw_col, r"[^0-9]", "")

    standardized_ssn = F.concat(F.substring(digits_values, 1, 3), F.lit("-"), F.substring(digits_values, 4, 2), F.lit("-"), F.substring(digits_values, 6, 4))

    invalid_condition = (raw_col.isNull() | F.lower(raw_col).isin(INVALID_STRINGS) | (F.length(digits_values) != 9))

    invalid_count = df.filter(invalid_condition).count()

    if verbose:
        if invalid_count > 0:
            invalid_values = (df.filter(invalid_condition).groupBy(column_name).count().orderBy(F.desc("count")).collect())
            invalid_values_list = [row[column_name] for row in invalid_values]
    
            print(f"[{column_name} column] First 20 invalid values: {invalid_values_list[:20]}")
            print(f">> {column_name} has {invalid_count} invalid values >> Converted to null")
        elif invalid_count == 0:
            print(f"[{column_name} column] This column doesn't have invalid values")
        print("-----")
    
    df = df.withColumn(column_name, F.when(invalid_condition, F.lit(None)).otherwise(standardized_ssn))
    
    return df
    
# ================================= Data processing for each tables =================================
def process_silver_table_loan_daily(snapshot_date_str, bronze_table_directory, silver_table_directory, spark, source_name, verbose=True):
    """
    Process the silver table for a given snapshot_date and source_name, and save the filtered data to a Parquet file.
    """
    silver_directory, filepath = get_silver_output_filepath(snapshot_date_str, silver_table_directory, source_name)
    if os.path.exists(filepath):
        if verbose:
            print("Skipped existing silver table:", filepath)
        return spark.read.parquet(filepath)

    df = load_bronze_table(snapshot_date_str, bronze_table_directory, spark, source_name, verbose=verbose)

    # -------------------------- start customized code for this table -----------------------
    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    float_cols_include_negative_values = ['balance']
    float_cols_exclude_negative_values = ['loan_amt', 'due_amt', 'paid_amt', 'overdue_amt'] 
    integer_cols = ['tenure', 'installment_num']
    string_cols = ['loan_id', 'Customer_ID']
           
    for column_name in float_cols_include_negative_values:
        df = convert_to_numeric_value(df, column_name, FloatType(), verbose=verbose)
    
    for column_name in float_cols_exclude_negative_values:
        df = convert_to_numeric_value(df, column_name, FloatType(), verbose=verbose)
        df = filter_value_range(df, column_name, min_value=0, verbose=verbose)
    
    for column_name in integer_cols:
        df = convert_to_numeric_value(df, column_name, IntegerType(), verbose=verbose)
        df = filter_value_range(df, column_name, min_value=0, verbose=verbose)

    for column_name in string_cols:
        df = clean_string_value(df, column_name, verbose=verbose)

    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
        else:
            raise ValueError(f"Column not found: {column}")

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn(
        "installments_missed",
        F.when(
            (col("due_amt").isNotNull()) & (col("due_amt") > 0) & (col("overdue_amt") > 0),
            F.ceil(col("overdue_amt") / col("due_amt"))
        ).otherwise(0).cast(IntegerType())
    )
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # -------------------------- end customized code for this table -----------------------

    # Create file_path 
    silver_directory = os.path.join(silver_table_directory, source_name)
    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
        
    # Create partition file name
    partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, partition_name)

    # Save to CSV
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df

def process_silver_table_features_attributes(snapshot_date_str, bronze_table_directory, silver_table_directory, spark, source_name, verbose=True):
    """ 
    Process the silver table for a given snapshot_date and source_name, and save the filtered data to a Parquet file.
    """
    silver_directory, filepath = get_silver_output_filepath(snapshot_date_str, silver_table_directory, source_name)
    if os.path.exists(filepath):
        if verbose:
            print("Skipped existing silver table:", filepath)
        return spark.read.parquet(filepath)

    df = load_bronze_table(snapshot_date_str, bronze_table_directory, spark, source_name, verbose=verbose)

    # -------------------------- start customized code for this table -----------------------
    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    string_cols = ['Customer_ID', 'Name', 'SSN', 'Occupation']
    
    for column_name in string_cols:
        df = clean_string_value(df, column_name, verbose=verbose)
    
    df = convert_to_numeric_value(df, "Age", IntegerType(), verbose=verbose)
    df = filter_value_range(df, "Age", min_value=18, max_value=120, verbose=verbose)
    df = standardize_ssn_value(df, verbose=verbose)
    
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
        else:
            raise ValueError(f"Column not found: {column}")

    # -------------------------- end customized code for this table -----------------------
    
    # Create file_path 
    silver_directory = os.path.join(silver_table_directory, source_name)
    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
        
    # Create partition file name
    partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, partition_name)

    # Save to CSV
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df

def process_silver_table_features_financials(snapshot_date_str, bronze_table_directory, silver_table_directory, spark, source_name, verbose=True):
    """ 
    Process the silver table for financial features. 
    """
    silver_directory, filepath = get_silver_output_filepath(snapshot_date_str, silver_table_directory, source_name)
    if os.path.exists(filepath):
        if verbose:
            print("Skipped existing silver table:", filepath)
        return spark.read.parquet(filepath)

    df = load_bronze_table(snapshot_date_str, bronze_table_directory, spark, source_name, verbose=verbose)

    # -------------------------- start customized code for this table -----------------------
    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    float_cols_include_negative_values = ['Changed_Credit_Limit', 'Monthly_Balance']
    float_cols_exclude_negative_values = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Outstanding_Debt'
                                          , 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly']
    integer_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']
    string_cols = ['Customer_ID', 'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour']
           
    for column_name in float_cols_include_negative_values:
        df = convert_to_numeric_value(df, column_name, FloatType(), verbose=verbose)
    
    for column_name in float_cols_exclude_negative_values:
        df = convert_to_numeric_value(df, column_name, FloatType(), verbose=verbose)
        df = filter_value_range(df, column_name, min_value=0, verbose=verbose)
    
    for column_name in integer_cols:
        df = convert_to_numeric_value(df, column_name, IntegerType(), verbose=verbose)
        df = filter_value_range(df, column_name, min_value=0, verbose=verbose)

    for column_name in string_cols:
        df = clean_string_value(df, column_name, verbose=verbose)

    df = convert_credit_history_age_to_months(df, "Credit_History_Age", verbose=verbose)
    
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType()
    }

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
        else:
            raise ValueError(f"Column not found: {column}")

    # Feature engineering: New Payment_Behavior column
    loan_clean = F.regexp_replace(F.col("Type_of_Loan"), r"(?i)\s*,?\s+and\s+", ",")
    df = df.withColumn("Loan_Type_Count", F.when(F.col("Type_of_Loan").isNull() | (F.trim(F.col("Type_of_Loan")) == "") | (F.col("Type_of_Loan") == "None"), 0) \
                                           .otherwise(F.size(F.split(loan_clean, r"\s*,\s*"))))

    # Feature engineering: New Credit_Mix_Score column
    df = df.withColumn("Credit_Mix_Score", F.when(F.col("Credit_Mix") == "Good", 3) \
                                            .when(F.col("Credit_Mix") == "Standard", 2) \
                                            .when(F.col("Credit_Mix") == "Bad", 1).otherwise(None))

    # Feature engineering: New Payment_of_Min_Amount_Score column
    df = df.withColumn("Payment_of_Min_Amount_Score", F.when(F.col("Payment_of_Min_Amount") == "Yes", 1) \
                                                       .when(F.col("Payment_of_Min_Amount") == "No", 0) \
                                                       .when(F.col("Payment_of_Min_Amount") == "NM", 0).otherwise(None))

    # Feature engineering: New Payment_Spending_Score column
    df = df.withColumn("Payment_Spending_Score", F.when(F.col("Payment_Behaviour").startswith("High_spent"), 2) \
                                                  .when(F.col("Payment_Behaviour").startswith("Low_spent"), 1) \
                                                  .otherwise(None))
                       
    # Feature engineering: New Payment_Value_Score column
    df = df.withColumn("Payment_Value_Score", F.when(F.col("Payment_Behaviour").contains("Large_value"), 3) \
                                               .when(F.col("Payment_Behaviour").contains("Medium_value"), 2) \
                                               .when(F.col("Payment_Behaviour").contains("Small_value"), 1) \
                                               .otherwise(None))
                       
    # -------------------------- end customized code for this table -----------------------
    
    # Create file_path 
    silver_directory = os.path.join(silver_table_directory, source_name)
    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
        
    # Create partition file name
    partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, partition_name)

    # Save to CSV
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df

def process_silver_table_features_clickstream(snapshot_date_str, bronze_table_directory, silver_table_directory, spark, source_name, verbose=True):
    """
    Process the silver table for clickstream features. 
    """
    df = load_bronze_table(snapshot_date_str, bronze_table_directory, spark, source_name, verbose=verbose)

    # -------------------------- start customized code for this table -----------------------
    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    fe_columns = [column for column in df.columns if column.startswith("fe_")]
    column_type_map = {column: IntegerType() for column in fe_columns}

    column_type_map.update({
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    })

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
        else:
            raise ValueError(f"Column not found: {column}")
            
    # Clickstream features are signed behavioural aggregates, so negative values are valid signals.
    # Do not null them out with a non-negative range filter.
    
    # -------------------------- end customized code for this table -----------------------
    
    # Create file_path 
    silver_directory = os.path.join(silver_table_directory, source_name)
    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
        
    # Create partition file name
    partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, partition_name)

    # Save to CSV
    df.write.mode("overwrite").parquet(filepath)
    if verbose:
        print('Saved to:', filepath)

    return df

source_processor = {
    'features_attributes': process_silver_table_features_attributes,
    'features_financials': process_silver_table_features_financials,
    'lms_loan_daily': process_silver_table_loan_daily,
    'feature_clickstream': process_silver_table_features_clickstream,
}

def get_silver_output_filepath(snapshot_date_str, silver_table_directory, source_name):
    """
    Get the file path for the silver table output. 
    """
    silver_directory = os.path.join(silver_table_directory, source_name)
    partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    return silver_directory, os.path.join(silver_directory, partition_name)

def process_silver_table_all(snapshot_date_str, bronze_table_directory, silver_table_directory, spark, source_name=None, verbose=True):
    """
    Process silver table based on source_name.
    """
    if source_name:
        process_function = source_processor.get(source_name)
        
        if process_function is None:
            raise ValueError(
                f"Invalid source_name: {source_name}. "
                f"Available options are: {list(source_processor.keys())}"
            )
        
        df = process_function(snapshot_date_str, bronze_table_directory, silver_table_directory, spark, source_name, verbose=verbose)
    else:
        all_source_name = ["features_attributes", "features_financials", "lms_loan_daily", "feature_clickstream"]
        for source_name_v2 in all_source_name:
            process_function = source_processor.get(source_name_v2)
            df = process_function(snapshot_date_str, bronze_table_directory, silver_table_directory, spark, source_name_v2, verbose=verbose)
        
    return df
    
