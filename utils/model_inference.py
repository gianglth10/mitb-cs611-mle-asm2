import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from utils.model_training import CANDIDATE_MODEL_NAMES, _assign_dataset_split


FEATURE_STORE_PATH = Path("datamart/gold/feature_store")
PREDICTION_PATH = Path("datamart/gold/model_predictions")
MODEL_BANK_PATH = Path("outputs/model_bank")


def _write_parquet_dir(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame to a Parquet directory, overwriting if it exists. 
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "part-00000.parquet", index=False)


def run_batch_inference() -> pd.DataFrame:
    """"
    Run batch inference using the champion model and save predictions to the gold table.
    """
    model = joblib.load(MODEL_BANK_PATH / "champion_model.pkl")
    with open(MODEL_BANK_PATH / "champion_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("model_name") not in CANDIDATE_MODEL_NAMES:
        raise ValueError(
            f"Unsupported or stale champion model '{metadata.get('model_name')}'. "
            "Run train_and_register_model() to refresh outputs/model_bank with the current candidates."
        )

    feature_store = pd.read_parquet(FEATURE_STORE_PATH)
    features = metadata["features"]
    scores = model.predict_proba(feature_store[features])[:, 1]
    threshold = float(metadata["decision_threshold"])

    predictions = feature_store[
        ["Customer_ID", "loan_id", "feature_snapshot_date", "label_snapshot_date"]
    ].copy()
    predictions["model_version"] = metadata["model_version"]
    predictions["dataset_split"] = _assign_dataset_split(feature_store, metadata)
    predictions["prediction_score"] = scores
    predictions["prediction_label"] = (scores >= threshold).astype(int)
    predictions["scored_at"] = datetime.now(timezone.utc).isoformat()

    _write_parquet_dir(predictions, PREDICTION_PATH)
    print(f"Saved gold/model_predictions: {len(predictions):,} rows")
    return predictions


if __name__ == "__main__":
    run_batch_inference()
