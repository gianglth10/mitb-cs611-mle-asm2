import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from utils.model_training import CANDIDATE_MODEL_NAMES


FEATURE_STORE_PATH = Path("datamart/gold/feature_store")
PREDICTION_PATH = Path("datamart/gold/model_predictions")
MONITORING_PATH = Path("datamart/gold/model_monitoring")
MODEL_BANK_PATH = Path("outputs/model_bank")
FIGURE_DIR = Path("outputs/monitoring/figures")
HEALTH_LOG_PATH = Path("outputs/monitoring/model_health_log.csv")

WARNING_AUC_DROP = 0.05
REVIEW_AUC_DROP = 0.10
WARNING_AUC_MIN = 0.75
REVIEW_AUC_MIN = 0.70


def _write_parquet_dir(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame to a parquet directory.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "part-00000.parquet", index=False)


def _safe_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """
    Calculate ROC AUC with error handling.
    """
    return float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan")


def _safe_average_precision(y_true: pd.Series, y_score: pd.Series) -> float:
    """
    Calculate average precision with error handling.
    """
    return float(average_precision_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan")


def _monitoring_frame(scored: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Create a monitoring frame from scored predictions and metadata.
    A monitoring frame is a summary of model performance metrics by month, including ROC AUC, average precision, precision, recall, F1 score, and confusion matrix counts.
    """
    rows = []
    for month, group in scored.groupby("monitoring_month"):
        y_true = group["label"].astype(int)
        y_score = group["prediction_score"]
        y_pred = group["prediction_label"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "monitoring_month": str(month),
                "model_version": metadata["model_version"],
                "row_count": int(len(group)),
                "actual_default_rate": float(y_true.mean()),
                "predicted_default_rate": float(y_pred.mean()),
                "roc_auc": _safe_roc_auc(y_true, y_score),
                "average_precision": _safe_average_precision(y_true, y_score),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows).sort_values("monitoring_month")


def _save_charts(scored: pd.DataFrame, monitoring: pd.DataFrame) -> None:
    """ 
    Save monitoring charts to outputs/monitoring/figures.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    auc_f1_df = monitoring.melt(
        id_vars=["monitoring_month"],
        value_vars=["roc_auc", "f1"],
        var_name="metric",
        value_name="score",
    )
    auc_f1_df["metric"] = auc_f1_df["metric"].map({"roc_auc": "ROC AUC", "f1": "F1 Score"})
    sns.lineplot(data=auc_f1_df, x="monitoring_month", y="score", hue="metric", marker="o")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Model ROC AUC and F1 by Feature Snapshot Month")
    plt.xlabel("Monitoring month")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "auc_f1_by_month.png", dpi=160)
    plt.savefig(FIGURE_DIR / "auc_by_month.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plot_df = monitoring.melt(
        id_vars=["monitoring_month"],
        value_vars=["actual_default_rate", "predicted_default_rate"],
        var_name="rate_type",
        value_name="rate",
    )
    sns.lineplot(data=plot_df, x="monitoring_month", y="rate", hue="rate_type", marker="o")
    plt.title("Actual vs Predicted Default Rate")
    plt.xlabel("Monitoring month")
    plt.ylabel("Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "default_rate_by_month.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    selected_months = list(scored["monitoring_month"].drop_duplicates().sort_values())
    if len(selected_months) > 6:
        selected_months = selected_months[:2] + selected_months[-4:]
    sns.histplot(
        data=scored[scored["monitoring_month"].isin(selected_months)],
        x="prediction_score",
        hue="monitoring_month",
        bins=20,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Prediction Score Distribution by Month")
    plt.xlabel("Prediction score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "prediction_score_distribution.png", dpi=160)
    plt.close()


def _metric_drop(baseline: float | None, latest: float | None) -> float | None:
    """
    Calculate the drop in a metric from baseline to latest.
    """
    if baseline is None or latest is None or pd.isna(baseline) or pd.isna(latest):
        return None
    return float(baseline - latest)


def _build_health_log(monitoring: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Create a lightweight governance record for model health review.
    """
    latest = monitoring.sort_values("monitoring_month").iloc[-1]
    oot_metrics = metadata.get("oot_metrics", {})
    baseline_auc = oot_metrics.get("roc_auc")
    latest_auc = latest.get("roc_auc")
    auc_drop = _metric_drop(baseline_auc, latest_auc)

    alert_reasons = []
    if latest_auc is not None and not pd.isna(latest_auc):
        if latest_auc < REVIEW_AUC_MIN:
            alert_reasons.append(f"latest_auc_below_{REVIEW_AUC_MIN:.2f}")
        elif latest_auc < WARNING_AUC_MIN:
            alert_reasons.append(f"latest_auc_below_{WARNING_AUC_MIN:.2f}")
    if auc_drop is not None:
        if auc_drop > REVIEW_AUC_DROP:
            alert_reasons.append(f"auc_drop_above_{REVIEW_AUC_DROP:.2f}")
        elif auc_drop > WARNING_AUC_DROP:
            alert_reasons.append(f"auc_drop_above_{WARNING_AUC_DROP:.2f}")

    if any(reason.startswith("latest_auc_below_0.70") or reason.startswith("auc_drop_above_0.10") for reason in alert_reasons):
        health_status = "Review Needed"
        champion_action = "Manual review before replacement"
    elif alert_reasons:
        health_status = "Warning"
        champion_action = "Monitor next run"
    else:
        health_status = "Healthy"
        champion_action = "Retain champion"

    report_path = Path("outputs/reports/assignment_2_slideument.pdf")
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "mlflow_registered_model_name": metadata.get("mlflow_registered_model_name"),
        "mlflow_registered_model_version": metadata.get("mlflow_registered_model_version"),
        "latest_monitoring_month": latest.get("monitoring_month"),
        "baseline_oot_auc": baseline_auc,
        "latest_auc": latest_auc,
        "auc_drop": auc_drop,
        "baseline_oot_recall": oot_metrics.get("recall"),
        "latest_recall": latest.get("recall"),
        "recall_drop": _metric_drop(oot_metrics.get("recall"), latest.get("recall")),
        "baseline_oot_f1": oot_metrics.get("f1"),
        "latest_f1": latest.get("f1"),
        "f1_drop": _metric_drop(oot_metrics.get("f1"), latest.get("f1")),
        "health_status": health_status,
        "alert_reason": "; ".join(alert_reasons) if alert_reasons else "none",
        "champion_action": champion_action,
        "monitoring_table_path": str(MONITORING_PATH),
        "auc_f1_chart_path": str(FIGURE_DIR / "auc_f1_by_month.png"),
        "default_rate_chart_path": str(FIGURE_DIR / "default_rate_by_month.png"),
        "score_distribution_chart_path": str(FIGURE_DIR / "prediction_score_distribution.png"),
        "report_path": str(report_path) if report_path.exists() else "",
        "retraining_automatic": False,
    }
    return pd.DataFrame([row])


def _append_health_log(health_log: pd.DataFrame) -> None:
    """
    Append health log entries to the existing health log file. 
    """
    HEALTH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HEALTH_LOG_PATH.exists():
        existing = pd.read_csv(HEALTH_LOG_PATH)
        health_log = pd.concat([existing, health_log], ignore_index=True)
    health_log.to_csv(HEALTH_LOG_PATH, index=False)


def run_model_monitoring() -> pd.DataFrame:
    """
    Run model monitoring and save results.
    """
    with open(MODEL_BANK_PATH / "champion_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("model_name") not in CANDIDATE_MODEL_NAMES:
        raise ValueError(
            f"Unsupported or stale champion model '{metadata.get('model_name')}'. "
            "Run train_and_register_model() before monitoring."
        )

    predictions = pd.read_parquet(PREDICTION_PATH)
    labels = pd.read_parquet(FEATURE_STORE_PATH).copy()

    for df in [predictions, labels]:
        df["feature_snapshot_date"] = pd.to_datetime(df["feature_snapshot_date"])
        df["label_snapshot_date"] = pd.to_datetime(df["label_snapshot_date"])

    scored = predictions.merge(
        labels,
        on=["Customer_ID", "loan_id", "feature_snapshot_date", "label_snapshot_date"],
        how="inner",
    )
    scored["monitoring_month"] = scored["feature_snapshot_date"].dt.to_period("M").astype(str)
    monitoring = _monitoring_frame(scored, metadata)
    _write_parquet_dir(monitoring, MONITORING_PATH)
    _save_charts(scored, monitoring)
    health_log = _build_health_log(monitoring, metadata)
    _append_health_log(health_log)
    print(f"Saved gold/model_monitoring: {len(monitoring):,} rows")
    print(
        "Model health status: "
        f"{health_log.iloc[-1]['health_status']} "
        f"({health_log.iloc[-1]['champion_action']})"
    )
    return monitoring


if __name__ == "__main__":
    run_model_monitoring()
