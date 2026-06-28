import csv
from datetime import datetime, timezone
from pathlib import Path


RUNTIME_LOG_PATH = Path("outputs/monitoring/dag_stage_runtime.csv")
RUNTIME_COLUMNS = [
    "recorded_at",
    "dag_id",
    "run_id",
    "task_id",
    "state",
    "try_number",
    "start_date",
    "end_date",
    "duration_seconds",
]


def record_task_runtime(context) -> None:
    """Append Airflow task runtime metadata for pipeline stage tracking."""
    task_instance = context["task_instance"]
    dag_run = context.get("dag_run")
    start_date = task_instance.start_date
    end_date = task_instance.end_date or datetime.now(timezone.utc)

    duration_seconds = None
    if start_date and end_date:
        duration_seconds = (end_date - start_date).total_seconds()

    row = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "dag_id": task_instance.dag_id,
        "run_id": dag_run.run_id if dag_run else None,
        "task_id": task_instance.task_id,
        "state": task_instance.state,
        "try_number": task_instance.try_number,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "duration_seconds": round(duration_seconds, 3) if duration_seconds is not None else None,
    }

    try:
        RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not RUNTIME_LOG_PATH.exists()
        with RUNTIME_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        print(f"Warning: could not write DAG runtime log to {RUNTIME_LOG_PATH}: {exc}")
