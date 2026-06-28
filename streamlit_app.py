from pathlib import Path

import pandas as pd
import streamlit as st


MONITORING_PATH = Path("datamart/gold/model_monitoring")
PREDICTION_PATH = Path("datamart/gold/model_predictions")
MODEL_METADATA_PATH = Path("outputs/model_bank/champion_metadata.json")
FIGURE_DIR = Path("outputs/monitoring/figures")
HEALTH_LOG_PATH = Path("outputs/monitoring/model_health_log.csv")


st.set_page_config(page_title="Loan Default Monitoring", layout="wide")
st.title("Loan Default Model Monitoring")


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


monitoring = _read_parquet(MONITORING_PATH)
predictions = _read_parquet(PREDICTION_PATH)
health_log = pd.read_csv(HEALTH_LOG_PATH) if HEALTH_LOG_PATH.exists() else None

if monitoring is None:
    st.info("Run the Airflow DAG or model monitoring step to generate monitoring outputs.")
    st.code("python -c \"from utils.model_monitoring import run_model_monitoring; run_model_monitoring()\"")
    st.stop()

latest = monitoring.sort_values("monitoring_month").iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Month", latest["monitoring_month"])
col2.metric("ROC AUC", f"{latest['roc_auc']:.3f}")
col3.metric("Recall", f"{latest['recall']:.3f}")
col4.metric("Rows", f"{int(latest['row_count']):,}")

if health_log is not None and not health_log.empty:
    latest_health = health_log.sort_values("created_at").iloc[-1]
    st.subheader("Champion Health Status")
    status_cols = st.columns(4)
    status_cols[0].metric("Health Status", latest_health["health_status"])
    status_cols[1].metric("Champion Action", latest_health["champion_action"])
    status_cols[2].metric("AUC Drop", f"{latest_health['auc_drop']:.3f}")
    status_cols[3].metric("Alert Reason", latest_health["alert_reason"])
    st.dataframe(health_log.sort_values("created_at", ascending=False), use_container_width=True)

st.subheader("Monthly Performance")
st.dataframe(monitoring.sort_values("monitoring_month"), use_container_width=True)

chart_cols = st.columns(3)
figures = [
    ("AUC and F1 by Month", FIGURE_DIR / "auc_f1_by_month.png"),
    ("Default Rate by Month", FIGURE_DIR / "default_rate_by_month.png"),
    ("Prediction Score Distribution", FIGURE_DIR / "prediction_score_distribution.png"),
]
for col, (title, path) in zip(chart_cols, figures):
    col.subheader(title)
    if path.exists():
        col.image(str(path), use_column_width=True)
    else:
        col.info(f"Missing figure: {path}")

if predictions is not None:
    st.subheader("Prediction Sample")
    st.dataframe(predictions.head(100), use_container_width=True)

if MODEL_METADATA_PATH.exists():
    st.subheader("Champion Model Metadata")
    st.json(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
