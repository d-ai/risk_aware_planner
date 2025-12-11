# src/logger.py
import csv
import os
from datetime import datetime
import json

LOG_DIR = "logs"
SUMMARY_FILENAME = "summary.csv"

SUMMARY_HEADER = [
    "run_id", "timestamp", "seed",
    "ego_x0", "ego_y0", "ego_v0",
    "n_other", "horizon_s", "dt_s",
    "traj_id", "traj_type",
    "collision_prob", "avg_min_distance", "worst_min_distance",
    "score", "chosen", "notes"
]


def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def init_summary_if_needed():
    """Create summary CSV with header if missing."""
    ensure_log_dir()
    summary_path = os.path.join(LOG_DIR, SUMMARY_FILENAME)
    if not os.path.exists(summary_path):
        with open(summary_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SUMMARY_HEADER)
    return summary_path


def append_summary_row(row_dict):
    """
    Append a single row (dict) to the summary CSV.
    Keys must match SUMMARY_HEADER; missing keys become empty strings.
    """
    summary_path = init_summary_if_needed()
    row = [row_dict.get(k, "") for k in SUMMARY_HEADER]

    with open(summary_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def make_run_id():
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S_%f")


def save_detail_json(run_id, detail_obj):
    """Save JSON lines file (one JSON object per line)."""
    ensure_log_dir()
    detail_path = os.path.join(LOG_DIR, f"detail_{run_id}.jsonl")
    with open(detail_path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(detail_obj) + "\n")
    return detail_path
