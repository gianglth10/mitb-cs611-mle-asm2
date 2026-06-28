# Loan Default MLE Pipeline

This repository contains an end-to-end machine learning engineering pipeline for loan default prediction. It demonstrates how raw loan and customer behavioural data can be processed into a reusable datamart, used to train and register a model, scored through batch inference, and monitored through downstream reporting.

The project is built to run locally with Docker Compose. It includes Airflow for orchestration, MLflow for experiment tracking and model artifacts, JupyterLab for notebook-based review, and Streamlit for the monitoring dashboard.

## Project Overview

**Business statement**

Financial institutions need to identify customers who are more likely to default so that credit risk teams can make better lending, pricing, and portfolio monitoring decisions. Manual rule-based checks are limited because they do not scale well across many behavioural, financial, and application attributes.

**Objective**

Build a reproducible MLE pipeline that predicts loan default risk using historical loan, financial, attribute, and clickstream data. The solution should support repeatable data processing, model training, model registration, batch inference, and model monitoring.

**Proposed solution**

The solution uses a layered datamart architecture:

```text
Raw data
  -> Bronze layer
  -> Silver layer
  -> Gold layer
  -> Model training and registration
  -> Batch inference
  -> Model monitoring
```

The bronze layer captures source snapshots with lineage metadata. The silver layer cleans and standardizes the source tables. The gold layer builds the label store and feature store. The model pipeline then trains candidate models, registers the champion model in MLflow, generates batch predictions, and produces monitoring outputs.

**Business impact**

This pipeline gives risk and business teams a repeatable way to score loan default risk, compare model performance over time, and monitor whether predictions remain reliable. In practice, this can support earlier intervention, better credit decisioning, improved portfolio oversight, and a clearer audit trail for model development.

## Project Structure

```text
.
|-- dags/
|   `-- loan_default_ml_pipeline_dag.py    Scheduled full pipeline DAG
|
|-- data/                                 Raw input CSV files
|   |-- lms_loan_daily.csv
|   |-- features_attributes.csv
|   |-- features_financials.csv
|   `-- feature_clickstream.csv
|
|-- datamart/                             Generated bronze, silver, and gold outputs
|-- notebooks/                            EDA and pipeline review notebooks
|-- outputs/                              Public run outputs and assignment artifacts
|   |-- model_bank/                       Champion model and model metadata
|   |-- mlflow/                           MLflow metadata and artifacts
|   |-- monitoring/figures/               Monitoring dashboard figures
|   |-- reports/                          Generated reports and slideuments
|   `-- runtime/                          Local Airflow/service logs, ignored by git
|
|-- utils/
|   |-- bronze_processing.py              Raw snapshot extraction and lineage
|   |-- silver_processing.py              Cleaning, typing, and standardization
|   |-- gold_processing.py                Label store and feature store creation
|   |-- datamart_full.py                  Datamart orchestration
|   |-- model_training.py                 Model training and MLflow registration
|   |-- model_inference.py                Batch inference
|   |-- model_monitoring.py               Model monitoring
|   `-- reporting.py                      Optional report generation utilities
|
|-- streamlit_app.py                      Monitoring dashboard
|-- docker-compose.yaml                   Local service orchestration
|-- Dockerfile                            Runtime image
|-- requirements.txt                      Python dependencies
`-- README.md
```

## Tech Stack

| Area | Tools |
| --- | --- |
| Orchestration | Apache Airflow |
| Data processing | Python, PySpark, Pandas, Parquet |
| Machine learning | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Dashboard | Streamlit |
| Notebook review | JupyterLab |
| Container runtime | Docker, Docker Compose |
| Visualization | Matplotlib, Seaborn |

## Services

When Docker Compose is running, the project exposes these local services:

| Service | URL | Purpose |
| --- | --- | --- |
| Airflow | http://localhost:8081 | Trigger and monitor pipeline DAGs |
| MLflow | http://localhost:5001 | View model experiments, metrics, and artifacts |
| JupyterLab | http://localhost:8888/lab | Run notebooks or terminal commands manually |
| ML monitoring dashboard | http://localhost:8501 | View monitoring output in Streamlit |

Airflow login:

```text
Username: airflow
Password: airflow
```

## How To Run

Run all commands from the project root:

```powershell
cd C:\Personal\Work\Github\mitb-cs611-mle-assignment-2
```

Build and start the services:

```powershell
docker compose build
docker compose up -d
```

Check that the containers are running:

```powershell
docker compose ps
```

Show the local service links:

```powershell
docker compose logs project-info
```

## Stage-By-Stage Run Guide

The recommended path is to run the full pipeline through Airflow. This validates the orchestration flow that connects the datamart, model training, inference, and monitoring stages.

The scheduled entry point is:

```text
loan_default_ml_pipeline
Schedule: daily at 10:00 AM Asia/Singapore time
Cron: 0 10 * * *
```

The DAG tracks runtime for each main stage and appends the results to:

```text
outputs/monitoring/dag_stage_runtime.csv
```

The runtime file includes DAG ID, run ID, task ID, state, start time, end time, and duration in seconds.

### Stage 0: Check DAG Imports

```powershell
docker compose exec airflow-scheduler airflow dags list-import-errors
```

Expected result:

```text
No data found
```

### Stage 1: Build Datamart

Trigger the full DAG manually:

```powershell
docker compose exec airflow-scheduler airflow dags trigger loan_default_ml_pipeline
```

This runs:

```text
loan_default_ml_pipeline
  build_bronze
  -> build_silver
  -> build_gold_label_and_feature_store
  -> train_and_register_model
  -> run_batch_inference
  -> run_model_monitoring
```

Datamart outputs:

```text
datamart/bronze/
datamart/silver/
datamart/gold/label_store/
datamart/gold/feature_store/
```

### Stage 2: Train And Register Model

The same DAG continues to:

```text
train_and_register_model
```

This stage trains Logistic Regression, Random Forest, and XGBoost candidates, evaluates them on train, test, and out-of-time splits, registers the champion model in MLflow, and saves public model artifacts under `outputs/`.

Model outputs:

```text
outputs/model_bank/champion_model.pkl
outputs/model_bank/champion_metadata.json
outputs/model_bank/candidate_metrics_latest.csv
outputs/model_bank/candidates/
outputs/model_bank/metrics_history.csv
outputs/mlflow/mlruns/
outputs/mlflow/mlartifacts/
```

Review the experiment in MLflow:

```text
http://localhost:5001
```

### Stage 3: Run Batch Inference

After model training, the same DAG continues to:

```text
run_batch_inference
```

This stage loads the champion model and scores the gold feature store.

Inference output:

```text
datamart/gold/model_predictions/
```

### Stage 4: Run Model Monitoring

After batch inference, the same DAG finishes with:

```text
run_model_monitoring
```

This stage creates model monitoring metrics and dashboard-ready figures.
It also writes a lightweight model health log that compares the latest monitoring month against the champion OOT baseline and flags the champion as `Healthy`, `Warning`, or `Review Needed`.

Monitoring outputs:

```text
datamart/gold/model_monitoring/
outputs/monitoring/model_health_log.csv
outputs/monitoring/figures/auc_f1_by_month.png
outputs/monitoring/figures/default_rate_by_month.png
outputs/monitoring/figures/prediction_score_distribution.png
```

The health log supports reliability controls and governance traceability:

```text
baseline OOT metrics vs latest monitoring metrics
metric drop thresholds and alert reason
champion action: Retain champion, Monitor next run, or Manual review before replacement
model version, monitoring paths, chart paths, and report path when available
```

Retraining is not automatic. Monitoring alerts are used to support manual review before a champion model is replaced.

Open the monitoring dashboard:

```text
http://localhost:8501
```

You can also open the generated chart files directly:

```text
outputs/monitoring/figures/auc_f1_by_month.png
outputs/monitoring/figures/default_rate_by_month.png
outputs/monitoring/figures/prediction_score_distribution.png
```

Optional PDF report generation is available through `utils/reporting.py`, but it is not part of the Airflow DAG:

```bash
python -c "from utils.reporting import generate_slideument; generate_slideument()"
```

Optional report output:

```text
outputs/reports/assignment_2_slideument.pdf
```

## Check DAG Status

Use these commands to check recent DAG runs:

```powershell
docker compose exec airflow-scheduler airflow dags list-runs -d loan_default_ml_pipeline --no-backfill
```

You can also monitor DAGs in the Airflow UI:

```text
http://localhost:8081
```

## Alternative Run Path: Jupyter Terminal

Use this path when you want to run each stage manually for checking or debugging.

Open JupyterLab:

```text
http://localhost:8888/lab
```

Open a Jupyter terminal, then run:

```bash
cd /opt/airflow/project
```

Run the datamart:

```bash
python -c "from utils.datamart_full import build_bronze, build_silver, build_gold_label_and_feature_store; build_bronze(); build_silver(); build_gold_label_and_feature_store()"
```

Train and register the model:

```bash
python -c "from utils.model_training import train_and_register_model; train_and_register_model()"
```

Run batch inference:

```bash
python -c "from utils.model_inference import run_batch_inference; run_batch_inference()"
```

Run monitoring:

```bash
python -c "from utils.model_monitoring import run_model_monitoring; run_model_monitoring()"
```

Run the full flow manually:

```bash
python -c "from utils.datamart_full import build_bronze, build_silver, build_gold_label_and_feature_store; from utils.model_training import train_and_register_model; from utils.model_inference import run_batch_inference; from utils.model_monitoring import run_model_monitoring; build_bronze(); build_silver(); build_gold_label_and_feature_store(); train_and_register_model(); run_batch_inference(); run_model_monitoring()"
```

## Notebooks

The `notebooks/` folder contains supporting notebooks for exploration and explanation:

```text
01_eda.ipynb
02_datamart_processing.ipynb
03_feature_engineering_before_model_training.ipynb
04_model_training_performance.ipynb
05_inference.ipynb
06_monitoring.ipynb
```

These notebooks are useful for reviewing the logic, but the production-style pipeline flow is executed through the Python modules and Airflow DAGs.

## Author And Citation

Author and course:

```text
Author: Giang Le (Stella)
Under supervise of: Prof Ulysses David Chong
Class: SMU MITB CS611 - Machine Learning Engineering
```
