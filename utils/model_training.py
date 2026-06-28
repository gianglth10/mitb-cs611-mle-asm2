import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


FEATURE_STORE_PATH = Path("datamart/gold/feature_store")
OUTPUT_PATH = Path("outputs")
MODEL_BANK_PATH = OUTPUT_PATH / "model_bank"
MLFLOW_EXPERIMENT_NAME = "loan_default_assignment_2"
MLFLOW_REGISTERED_MODEL_NAME = "loan_default_champion"
CANDIDATE_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost"}

ID_COLUMNS = {
    "Customer_ID",
    "loan_id",
    "label",
    "label_def",
    "feature_snapshot_date",
    "label_snapshot_date",
}


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    """
    Calculate ROC AUC with error handling. Returns NaN if y_true has only one class.
    """
    return float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan")


def _safe_average_precision(y_true: pd.Series, y_score: np.ndarray) -> float:
    """
    Calculate average precision with error handling. Returns NaN if y_true has only one class.
    """
    return float(average_precision_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan")


def _metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    """ 
    Calculate various classification metrics based on true labels, predicted scores, and a decision threshold.
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "average_precision": _safe_average_precision(y_true, y_score),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _best_threshold(y_true: pd.Series, y_score: np.ndarray) -> float:
    """
    Find the optimal decision threshold based on the highest F1 score.
    """
    candidates = np.linspace(0.05, 0.95, 19)
    scores = [f1_score(y_true, y_score >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def _json_safe(value):
    """
    Make a value JSON-safe.
    """
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """ 
    Get the list of feature column names from a DataFrame.
   """
    return [col for col in df.columns if col not in ID_COLUMNS]


def _setup_mlflow() -> None:
    """ 
    Set up MLflow tracking.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str((OUTPUT_PATH / "mlflow" / "mlruns").resolve()))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    """ 
    Build a preprocessor for the given features.
    """
    numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in features if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train, test, and out-of-time (OOT) sets based on feature snapshot dates. 
    """
    df = df.sort_values("feature_snapshot_date").reset_index(drop=True)
    dates = sorted(pd.to_datetime(df["feature_snapshot_date"]).dropna().unique())
    if len(dates) < 3:
        train_end_idx = max(1, int(len(df) * 0.6))
        test_end_idx = max(train_end_idx + 1, int(len(df) * 0.8))
        return (
            df.iloc[:train_end_idx].copy(),
            df.iloc[train_end_idx:test_end_idx].copy(),
            df.iloc[test_end_idx:].copy(),
        )

    train_cutoff = dates[max(1, int(len(dates) * 0.6))]
    oot_cutoff = dates[max(2, int(len(dates) * 0.8))]
    train_df = df[df["feature_snapshot_date"] < train_cutoff].copy()
    test_df = df[(df["feature_snapshot_date"] >= train_cutoff) & (df["feature_snapshot_date"] < oot_cutoff)].copy()
    oot_df = df[df["feature_snapshot_date"] >= oot_cutoff].copy()
    if train_df.empty or test_df.empty or oot_df.empty:
        train_end_idx = max(1, int(len(df) * 0.6))
        test_end_idx = max(train_end_idx + 1, int(len(df) * 0.8))
        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[train_end_idx:test_end_idx].copy()
        oot_df = df.iloc[test_end_idx:].copy()
    return train_df, test_df, oot_df


def _split_window(df: pd.DataFrame) -> dict:
    """ 
    Get the start and end dates of the feature snapshot date in the DataFrame.
    """
    if df.empty:
        return {"start_date": None, "end_date": None, "rows": 0}
    return {
        "start_date": df["feature_snapshot_date"].min().date().isoformat(),
        "end_date": df["feature_snapshot_date"].max().date().isoformat(),
        "rows": int(len(df)),
    }


def _assign_dataset_split(df: pd.DataFrame, metadata: dict) -> pd.Series:
    """ 
    Assign each row to a dataset split based on its feature snapshot date and the specified split windows.
    """
    feature_dates = pd.to_datetime(df["feature_snapshot_date"])
    if "split_windows" in metadata:
        train_end = pd.Timestamp(metadata["split_windows"]["train"]["end_date"])
        test_end = pd.Timestamp(metadata["split_windows"]["test"]["end_date"])
    else:
        train_end = pd.Timestamp(metadata["training_end_date"])
        test_end = pd.Timestamp(metadata.get("test_end_date") or metadata.get("validation_end_date"))
    return np.select(
        [feature_dates <= train_end, feature_dates <= test_end],
        ["train", "test"],
        default="oot",
    )


def _print_performance_table(performance_rows: list[dict]) -> None:
    """
    Print a formatted table of model performance metrics. 
    """
    table = pd.DataFrame(performance_rows)
    display_columns = [
        "model_name",
        "dataset_split",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "decision_threshold",
        "row_count",
        "default_rate",
    ]
    table = table[display_columns].sort_values(["model_name", "dataset_split"])
    metric_columns = ["roc_auc", "average_precision", "precision", "recall", "f1", "decision_threshold", "default_rate"]
    table[metric_columns] = table[metric_columns].round(4)
    print("\nModel performance by split:")
    print(table.to_string(index=False))


def train_and_register_model() -> dict:
    """ 
    Train candidate models, evaluate their performance, and register the best-performing model as the champion in MLflow.
    """
    _setup_mlflow()
    feature_store = pd.read_parquet(FEATURE_STORE_PATH)
    feature_store["feature_snapshot_date"] = pd.to_datetime(feature_store["feature_snapshot_date"])
    feature_store["label_snapshot_date"] = pd.to_datetime(feature_store["label_snapshot_date"])
    feature_store = feature_store.dropna(subset=["label"]).copy()
    feature_store["label"] = feature_store["label"].astype(int)

    features = _feature_columns(feature_store)
    train_df, test_df, oot_df = _time_split(feature_store)
    X_train = train_df[features]
    y_train = train_df["label"]
    X_test = test_df[features]
    y_test = test_df["label"]
    X_oot = oot_df[features]
    y_oot = oot_df["label"]
    model_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    preprocessor = _build_preprocessor(feature_store, features)
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = {}
    train_results = {}
    oot_results = {}
    fitted_models = {}
    mlflow_run_ids = {}
    performance_rows = []
    for model_name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        train_scores = pipeline.predict_proba(X_train)[:, 1]
        test_scores = pipeline.predict_proba(X_test)[:, 1]
        oot_scores = pipeline.predict_proba(X_oot)[:, 1]
        threshold = _best_threshold(y_test, test_scores)
        train_results[model_name] = _metrics(y_train, train_scores, threshold)
        train_results[model_name]["decision_threshold"] = threshold
        results[model_name] = _metrics(y_test, test_scores, threshold)
        results[model_name]["decision_threshold"] = threshold
        oot_results[model_name] = _metrics(y_oot, oot_scores, threshold)
        oot_results[model_name]["decision_threshold"] = threshold
        fitted_models[model_name] = pipeline
        for split_name, split_metrics, split_df in [
            ("train", train_results[model_name], train_df),
            ("test", results[model_name], test_df),
            ("oot", oot_results[model_name], oot_df),
        ]:
            performance_rows.append(
                {
                    "model_name": model_name,
                    "dataset_split": split_name,
                    "row_count": int(len(split_df)),
                    "default_rate": float(split_df["label"].mean()),
                    **split_metrics,
                }
            )

        with mlflow.start_run(run_name=f"{model_version}_{model_name}") as run:
            mlflow_run_ids[model_name] = run.info.run_id
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("stage", "candidate")
            mlflow.set_tag("champion", "false")
            mlflow.log_param("training_start_date", train_df["feature_snapshot_date"].min().date().isoformat())
            mlflow.log_param("training_end_date", train_df["feature_snapshot_date"].max().date().isoformat())
            mlflow.log_param("test_start_date", test_df["feature_snapshot_date"].min().date().isoformat())
            mlflow.log_param("test_end_date", test_df["feature_snapshot_date"].max().date().isoformat())
            mlflow.log_param("oot_start_date", oot_df["feature_snapshot_date"].min().date().isoformat())
            mlflow.log_param("oot_end_date", oot_df["feature_snapshot_date"].max().date().isoformat())
            mlflow.log_param("feature_count", len(features))
            mlflow.log_param("training_rows", len(train_df))
            mlflow.log_param("test_rows", len(test_df))
            mlflow.log_param("oot_rows", len(oot_df))
            mlflow.log_param("decision_threshold", threshold)
            for param_name, param_value in estimator.get_params(deep=False).items():
                mlflow.log_param(f"model__{param_name}", param_value)
            mlflow.log_metrics({f"train_{key}": value for key, value in train_results[model_name].items()})
            mlflow.log_metrics({f"test_{key}": value for key, value in results[model_name].items()})
            mlflow.log_metrics({f"oot_{key}": value for key, value in oot_results[model_name].items()})
            mlflow.log_metrics(results[model_name])
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

    _print_performance_table(performance_rows)

    def ranking_key(item: tuple[str, dict]) -> float:
        metrics = item[1]
        roc_auc = metrics["roc_auc"]
        return -1.0 if roc_auc is None or np.isnan(roc_auc) else roc_auc

    champion_name, champion_metrics = max(results.items(), key=ranking_key)
    champion_oot_metrics = oot_results[champion_name]
    champion = fitted_models[champion_name]

    client = MlflowClient()
    for model_name, run_id in mlflow_run_ids.items():
        client.set_tag(run_id, "champion", str(model_name == champion_name).lower())
        client.set_tag(run_id, "stage", "champion" if model_name == champion_name else "challenger")

    champion_run_id = mlflow_run_ids[champion_name]
    registered_model = mlflow.register_model(
        model_uri=f"runs:/{champion_run_id}/model",
        name=MLFLOW_REGISTERED_MODEL_NAME,
    )
    client.set_model_version_tag(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="model_version",
        value=model_version,
    )
    client.set_model_version_tag(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="model_name",
        value=champion_name,
    )
    client.set_registered_model_alias(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    MODEL_BANK_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, MODEL_BANK_PATH / "champion_model.pkl")
    candidate_model_path = MODEL_BANK_PATH / "candidates"
    candidate_model_path.mkdir(parents=True, exist_ok=True)
    for model_name, fitted_model in fitted_models.items():
        joblib.dump(fitted_model, candidate_model_path / f"{model_version}_{model_name}.pkl")

    metadata = {
        "model_name": champion_name,
        "model_version": model_version,
        "mlflow_run_id": champion_run_id,
        "mlflow_registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
        "mlflow_registered_model_version": registered_model.version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_start_date": train_df["feature_snapshot_date"].min().date().isoformat(),
        "training_end_date": train_df["feature_snapshot_date"].max().date().isoformat(),
        "test_start_date": test_df["feature_snapshot_date"].min().date().isoformat(),
        "test_end_date": test_df["feature_snapshot_date"].max().date().isoformat(),
        "oot_start_date": oot_df["feature_snapshot_date"].min().date().isoformat(),
        "oot_end_date": oot_df["feature_snapshot_date"].max().date().isoformat(),
        "validation_start_date": test_df["feature_snapshot_date"].min().date().isoformat(),
        "validation_end_date": test_df["feature_snapshot_date"].max().date().isoformat(),
        "split_strategy": "time_based_60_20_20_by_feature_snapshot_date",
        "split_windows": {
            "train": _split_window(train_df),
            "test": _split_window(test_df),
            "oot": _split_window(oot_df),
        },
        "features": features,
        "row_counts": {
            "training": int(len(train_df)),
            "test": int(len(test_df)),
            "oot": int(len(oot_df)),
            "validation": int(len(test_df)),
        },
        "class_balance": {
            "training_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
            "oot_default_rate": float(y_oot.mean()),
            "validation_default_rate": float(y_test.mean()),
        },
        "candidate_metrics": results,
        "candidate_train_metrics": train_results,
        "candidate_test_metrics": results,
        "candidate_oot_metrics": oot_results,
        "metrics": champion_metrics,
        "oot_metrics": champion_oot_metrics,
        "decision_threshold": champion_metrics["decision_threshold"],
    }
    with open(MODEL_BANK_PATH / "champion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)

    metrics_history = pd.DataFrame(
        [
            {
                "model_version": model_version,
                "model_name": name,
                "dataset_split": "train",
                **metrics,
            }
            for name, metrics in train_results.items()
        ]
        + [
            {
                "model_version": model_version,
                "model_name": name,
                "dataset_split": "test",
                **metrics,
            }
            for name, metrics in results.items()
        ]
        + [
            {
                "model_version": model_version,
                "model_name": name,
                "dataset_split": "oot",
                **metrics,
            }
            for name, metrics in oot_results.items()
        ]
    )
    metrics_history["is_champion"] = metrics_history["model_name"] == champion_name
    metrics_history.to_csv(MODEL_BANK_PATH / "candidate_metrics_latest.csv", index=False)

    history_path = MODEL_BANK_PATH / "metrics_history.csv"
    if history_path.exists():
        prior = pd.read_csv(history_path)
        prior = prior[prior["model_name"].isin(CANDIDATE_MODEL_NAMES)]
        metrics_history = pd.concat([prior, metrics_history], ignore_index=True)
    metrics_history.to_csv(history_path, index=False)

    print(f"Registered champion model {champion_name} version {model_version}")
    return metadata


if __name__ == "__main__":
    train_and_register_model()
