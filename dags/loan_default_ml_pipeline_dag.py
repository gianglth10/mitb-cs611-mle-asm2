import os
import sys
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/airflow/project"))
if PROJECT_DIR.exists():
    os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.dag_runtime import record_task_runtime


default_args = {"owner": "cs611", "retries": 0}
LOCAL_TZ = pendulum.timezone("Asia/Singapore")
runtime_callbacks = {
    "on_success_callback": record_task_runtime,
    "on_failure_callback": record_task_runtime,
}


def build_bronze_stage() -> None:
    from utils.datamart_full import build_bronze

    build_bronze()


def build_silver_stage() -> None:
    from utils.datamart_full import build_silver

    build_silver()


def build_gold_label_and_feature_store_stage() -> None:
    from utils.datamart_full import build_gold_label_and_feature_store

    build_gold_label_and_feature_store()


def train_and_register_model_stage() -> None:
    from utils.model_training import train_and_register_model

    train_and_register_model()


def run_batch_inference_stage() -> None:
    from utils.model_inference import run_batch_inference

    run_batch_inference()


def run_model_monitoring_stage() -> None:
    from utils.model_monitoring import run_model_monitoring

    run_model_monitoring()


with DAG(
    dag_id="loan_default_ml_pipeline",
    description="Train, score, and monitor a loan default prediction model.",
    default_args=default_args,
    start_date=pendulum.datetime(2023, 1, 1, tz=LOCAL_TZ),
    schedule="0 10 * * *",
    catchup=False,
    tags=["assignment_2", "loan_default", "ml_pipeline"],
) as dag:
    start = EmptyOperator(task_id="start")

    build_bronze_task = PythonOperator(task_id="build_bronze", python_callable=build_bronze_stage, **runtime_callbacks)
    build_silver_task = PythonOperator(task_id="build_silver", python_callable=build_silver_stage, **runtime_callbacks)
    build_gold_label_and_feature_store_task = PythonOperator(
        task_id="build_gold_label_and_feature_store",
        python_callable=build_gold_label_and_feature_store_stage,
        **runtime_callbacks,
    )
    train_model_task = PythonOperator(
        task_id="train_and_register_model",
        python_callable=train_and_register_model_stage,
        **runtime_callbacks,
    )
    inference_task = PythonOperator(task_id="run_batch_inference", python_callable=run_batch_inference_stage, **runtime_callbacks)
    monitoring_task = PythonOperator(task_id="run_model_monitoring", python_callable=run_model_monitoring_stage, **runtime_callbacks)
    end = EmptyOperator(task_id="end")

    start >> build_bronze_task >> build_silver_task >> build_gold_label_and_feature_store_task >> train_model_task >> inference_task >> monitoring_task >> end
