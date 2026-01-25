"""
Phase 0 - Window Count Audit
Counts number of sliding windows per patient using the training pipeline settings.
Outputs: docs/reports/ensemble_v5/window_stats.csv
"""

import csv
from pathlib import Path
import sys
import statistics

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ml.training.train_v4_ensemble import (
    CTUCHBDataLoader,
    generate_sliding_windows,
    load_config,
)

DATA_DIR = PROJECT_ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "reports" / "ensemble_v5" / "window_stats.csv"
FS_HZ = 4.0


def main():
    config = load_config()
    window_cfg = config.get("windowing", {})
    window_minutes = window_cfg.get("window_minutes", 20)
    stride_minutes = window_cfg.get("stride_minutes", 5)
    window_samples = int(window_minutes * 60 * FS_HZ)
    stride_samples = int(stride_minutes * 60 * FS_HZ)

    loader = CTUCHBDataLoader(DATA_DIR)
    fhr_signals, _, patient_ids = loader.load_all_records()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    max_windows = 0
    min_windows = float("inf")
    counts = []

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "num_windows"])
        for fhr, pid in zip(fhr_signals, patient_ids):
            windows = generate_sliding_windows(fhr, window_samples, stride_samples)
            window_count = len(windows)
            writer.writerow([pid, window_count])
            total_windows += window_count
            max_windows = max(max_windows, window_count)
            min_windows = min(min_windows, window_count)
            counts.append(window_count)

    print(f"Window stats written to {OUTPUT_PATH}")
    print(
        f"Params: window={window_minutes} min ({window_samples} samples), "
        f"stride={stride_minutes} min ({stride_samples} samples)"
    )
    num_patients = len(patient_ids)
    mean_windows = statistics.mean(counts) if counts else 0
    median_windows = statistics.median(counts) if counts else 0
    min_windows = min_windows if counts else 0

    print("| Metric | Value |")
    print("| --- | --- |")
    print(f"| Num patients | {num_patients} |")
    print(f"| Total windows | {total_windows} |")
    print(f"| Mean windows / patient | {mean_windows:.2f} |")
    print(f"| Median windows / patient | {median_windows:.2f} |")
    print(f"| Min windows / patient | {min_windows} |")
    print(f"| Max windows / patient | {max_windows} |")


if __name__ == "__main__":
    main()
