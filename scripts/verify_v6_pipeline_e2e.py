"""End-to-end verification for V6 training pack (Pre-AI only)."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
import json
import zipfile
from io import BytesIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime_config import load_runtime_config, apply_strict_warnings
from src.v6.pre_ai.pipeline import run_pre_ai


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase5_verify_v6_2.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify V6 training pack end-to-end (Pre-AI)")
    parser.add_argument("--pack", default="training_pack_v6_2.zip", help="Path to training pack zip")
    parser.add_argument("--log-dir", default=None, help="Override log directory")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else Path("REPORTS") / "audit_artifacts" / _timestamp()
    log_path = _configure_logging(log_dir)

    cfg = load_runtime_config()
    apply_strict_warnings(cfg.strict_mode)

    pack_path = Path(args.pack)
    if not pack_path.exists():
        logging.error("Pack not found: %s", pack_path)
        return 1

    with zipfile.ZipFile(pack_path, "r") as zf:
        try:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        except Exception as exc:
            logging.error("Failed to read manifest.json: %s", exc)
            return 1

        patients = manifest.get("patients", [])
        datasets = ["CTU-CHB", "CTGDL", "FHRMA"]

        overall_ok = True
        for dataset in datasets:
            included = [p for p in patients if p.get("source") == dataset and p.get("included")]
            if not included:
                skips = [p.get("skip_reason") for p in patients if p.get("source") == dataset]
                top = {}
                for reason in skips:
                    if not reason:
                        continue
                    top[reason] = top.get(reason, 0) + 1
                top_sorted = sorted(top.items(), key=lambda x: x[1], reverse=True)
                logging.info("NOT FOUND: %s (top skips: %s)", dataset, top_sorted[:3])
                overall_ok = False
                continue

            chosen = random.choice(included)
            npz_name = f"records/{chosen.get('patient_id')}.npz"
            logging.info("Selected record for %s: %s", dataset, npz_name)
            with zf.open(npz_name) as f:
                data = np.load(BytesIO(f.read()))

            fs_hz = float(np.asarray(data["fs_hz"]).item())
            patient_id = str(np.asarray(data["patient_id"]).item())
            fhr_raw = np.asarray(data["fhr_raw"], dtype=float)
            uc_raw = np.asarray(data["uc_raw"], dtype=float)

            if fs_hz != cfg.fs_hz:
                logging.error("[%s] FS mismatch: %s != %s", dataset, fs_hz, cfg.fs_hz)
                overall_ok = False
                continue
            if len(fhr_raw) != len(uc_raw):
                logging.error("[%s] Alignment fail: fhr=%d uc=%d", dataset, len(fhr_raw), len(uc_raw))
                overall_ok = False
                continue

            records = run_pre_ai(patient_id, fhr_raw, uc_raw)
            if not records:
                logging.error("[%s] No windows produced", dataset)
                overall_ok = False
                continue

            for rec in records:
                minutes = rec.get("window_minutes")
                if minutes is None:
                    logging.error("[%s] Missing window_minutes", dataset)
                    overall_ok = False
                    break
                if abs(float(minutes) - cfg.window_minutes) > 0.01:
                    logging.error("[%s] Window size mismatch: %s", dataset, minutes)
                    overall_ok = False
                    break
                qc = rec.get("quality_class")
                if qc not in {"HIGH", "MED", "LOW"}:
                    logging.error("[%s] Invalid quality_class: %s", dataset, qc)
                    overall_ok = False
                    break

            if overall_ok:
                logging.info("GREEN LIGHT: %s passed (%d windows)", dataset, len(records))

    if overall_ok:
        logging.info("GREEN LIGHT: E2E verification passed for all datasets")
        logging.info("Log file: %s", log_path)
        return 0
    logging.error("E2E verification failed for one or more datasets")
    logging.info("Log file: %s", log_path)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
