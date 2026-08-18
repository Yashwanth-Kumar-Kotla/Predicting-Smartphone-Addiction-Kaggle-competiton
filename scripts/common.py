"""Shared utilities: data loading, CV folds, experiment logging."""
import csv
import os
import pandas as pd

DATA_DIR = "playground-series-s6e8"
TARGET = "addicted_label"
ID_COL = "id"
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
NUM_COLS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
FEATURE_COLS = NUM_COLS + CAT_COLS
N_FOLDS = 5
SEED = 42

LOG_PATH = "experiments/experiment_log.csv"
LOG_FIELDS = [
    "exp_id", "model", "features", "preprocessing", "hyperparams",
    "cv_strategy", "cv_mean", "cv_std", "best_fold", "worst_fold",
    "runtime_sec", "notes", "conclusion",
]


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
    return train, test


def log_experiment(row: dict):
    """Append one row to the persistent experiment log, creating it if needed."""
    file_exists = os.path.exists(LOG_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})
