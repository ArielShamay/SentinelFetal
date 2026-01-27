"""Build training_pack_v6.zip from standardized ingest (Pre-AI only)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
from hashlib import sha1
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime_config import load_runtime_config, apply_strict_warnings
from src.v6.pre_ai.invariants import assert_raw_invariants, WarmupError
from src.v6.pre_ai.pipeline import run_pre_ai
from src.v6.pre_ai.quality_policy import load_quality_policy
from src.v6.pre_ai.ingest.schema import record_reject_reason
from src.v6.pre_ai.ingest.ctu_loader import discover_ctu_records, load_ctu_record
from src.v6.pre_ai.ingest.ctgd_loader import load_ctgdl_record
from src.v6.pre_ai.ingest.fhrma_loader import discover_fhrma_records, load_fhrma_record


LOGGER = logging.getLogger("build_v6_training_pack")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _get_git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _ensure_log_dir(log_dir: Path | None) -> Path:
    if log_dir is None:
        log_dir = Path("REPORTS") / "audit_artifacts" / _timestamp()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _configure_logging(log_dir: Path) -> Path:
    log_path = log_dir / "build_v6_training_pack.log"
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    return log_path


def _find_dataset_root(data_root: Path, candidates: List[str]) -> Path | None:
    for name in candidates:
        candidate = data_root / name
        if candidate.exists():
            return candidate
    for child in data_root.iterdir() if data_root.exists() else []:
        if not child.is_dir():
            continue
        lowered = child.name.lower()
        if any(token.lower() in lowered for token in candidates):
            return child
    return None


def _save_npz(out_path: Path, record) -> None:
    labels_json = json.dumps(record.labels, ensure_ascii=True)
    meta_json = json.dumps(record.meta, ensure_ascii=True)
    quality_json = json.dumps(record.record_quality, ensure_ascii=True)
    fhr_filled = record.fhr_filled if record.fhr_filled is not None else record.fhr_raw
    uc_filled = record.uc_filled if record.uc_filled is not None else record.uc_raw
    np.savez_compressed(
        out_path,
        fhr_raw=record.fhr_raw,
        uc_raw=record.uc_raw,
        fhr_filled=fhr_filled,
        uc_filled=uc_filled,
        fs_hz=np.array(record.fs_hz),
        patient_id=np.array(record.patient_id),
        source=np.array(record.source),
        labels_json=np.array(labels_json),
        meta_json=np.array(meta_json),
        record_quality_json=np.array(quality_json),
    )


def _write_summary(manifest: Dict, report_path: Path) -> None:
    stats = manifest.get("stats", {})
    counts = stats.get("counts", {})
    skipped = stats.get("skipped_by_reason", {})
    durations = stats.get("duration_minutes", {})
    quality = stats.get("quality_class_counts", {})
    counts_by_dataset = stats.get("counts_by_dataset", {})
    counts_by_task = stats.get("counts_by_target_task", {})

    lines = [
        "# V6 Dataset Summary (Pre-AI) — v6.2",
        "",
        f"Build timestamp: {manifest.get('build_timestamp')}",
        f"Commit: {manifest.get('git_commit')}",
        "",
        "## Counts",
        f"- total_records: {counts.get('total_records', 0)}",
        f"- included: {counts.get('included', 0)}",
        f"- skipped: {counts.get('skipped', 0)}",
        "",
        "## Counts by dataset",
    ]
    if counts_by_dataset:
        for name, vals in counts_by_dataset.items():
            lines.append(
                f"- {name}: total={vals.get('total', 0)} "
                f"included={vals.get('included', 0)} skipped={vals.get('skipped', 0)}"
            )
    else:
        lines.append("- none")

    lines += [
        "",
        "## Counts by target_task",
    ]
    if counts_by_task:
        for name, val in counts_by_task.items():
            lines.append(f"- {name}: {val}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Skipped by reason",
    ]
    if skipped:
        for reason, count in skipped.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Duration (minutes)",
        f"- min: {durations.get('min', 0):.2f}",
        f"- mean: {durations.get('mean', 0):.2f}",
        f"- max: {durations.get('max', 0):.2f}",
        "",
        "## Quality class distribution (windows)",
    ]
    if quality:
        for key, val in quality.items():
            lines.append(f"- {key}: {val}")
    else:
        lines.append("- none")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_fhrma_forensics(report_path: Path, root: Path | None) -> None:
    lines = ["# FHRMA Forensics (V6.2)", ""]
    if root is None or not root.exists():
        lines.append("Dataset root not found.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(f"Resolved root: {root}")
    lines.append("")
    ext_counts: Dict[str, int] = {}
    samples: Dict[str, list[str]] = {}
    dir_stats: list[tuple[int, str, int]] = []
    max_depth = 0
    for dirpath, _, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        max_depth = max(max_depth, depth)
        dir_stats.append((depth, str(Path(dirpath)), len(filenames)))
        for name in filenames:
            ext = Path(name).suffix.lower() or "<none>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            samples.setdefault(ext, [])
            if len(samples[ext]) < 10:
                samples[ext].append(str(Path(dirpath) / name))

    lines.append(f"Max depth: {max_depth}")
    lines.append("")
    lines.append("## Directory tree summary (depth, path, file_count)")
    for depth, path, count in sorted(dir_stats, key=lambda x: (x[0], x[1]))[:200]:
        lines.append(f"- depth={depth} files={count} path={path}")
    lines.append("")
    lines.append("## Counts by extension")
    for ext, count in sorted(ext_counts.items(), key=lambda x: x[0]):
        lines.append(f"- {ext}: {count}")

    lines.append("")
    lines.append("## Sample files (up to 10 per extension)")
    for ext, items in sorted(samples.items(), key=lambda x: x[0]):
        lines.append(f"- {ext}:")
        for item in items:
            lines.append(f"  - {item}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_ctgdl_suggestions(report_path: Path, entries: list[dict]) -> None:
    lines = ["# CTGDL Column Suggestions (V6.2)", ""]
    if not entries:
        lines.append("No suggestions captured.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for entry in entries:
        lines.append(f"## File: {entry.get('path')}")
        cols = entry.get("columns", [])
        lines.append(f"- Columns: {', '.join(cols)}")
        lines.append("- Top candidates:")
        for cand in entry.get("top_candidates", []):
            lines.append(
                f"  - {cand.get('column')} | in_range={cand.get('in_range_frac'):.3f} "
                f"finite={cand.get('finite_frac'):.3f} "
                f"min={cand.get('min')} med={cand.get('median')} max={cand.get('max')}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _safe_extract_tar(archive_path: Path, dest_root: Path) -> Tuple[Path, bool]:
    name = archive_path.name
    if name.endswith(".tar.gz"):
        name = name[: -len(".tar.gz")]
    elif name.endswith(".tgz"):
        name = name[: -len(".tgz")]
    digest = sha1(str(archive_path).encode("utf-8")).hexdigest()[:8]
    dest_dir = dest_root / f"{name}_{digest}"
    marker = dest_dir / ".extracted.ok"
    if marker.exists():
        return dest_dir, False

    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = False
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_name = member.name
            if member_name.startswith("/") or ".." in Path(member_name).parts:
                continue
            target = dest_dir / member_name
            target_parent = target.parent.resolve()
            if not str(target_parent).startswith(str(dest_dir.resolve())):
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src:
                if src is None:
                    continue
                with open(target, "wb") as dst:
                    dst.write(src.read())
                extracted = True
    if extracted:
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    return dest_dir, extracted


def _inspect_tar_members(archive_path: Path, max_members: int = 20) -> dict:
    members = []
    csv_members = []
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                name = member.name
                members.append(name)
                if name.lower().endswith(".csv"):
                    csv_members.append(name)
                if len(members) >= max_members and len(csv_members) >= max_members:
                    break
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "members": members[:max_members],
        "csv_members": csv_members[:max_members],
    }


def _write_ctgdl_extraction_report(
    report_path: Path,
    archives: list[Path],
    inspected: dict[Path, dict],
    extracted_dirs: list[Path],
) -> None:
    lines = ["# CTGDL Extraction Report (V6.2)", ""]
    if not archives:
        lines.append("No archives found.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(f"Archives found: {len(archives)}")
    lines.append("")
    for archive in archives:
        size_mb = archive.stat().st_size / (1024 * 1024)
        lines.append(f"- {archive} ({size_mb:.2f} MB)")
    lines.append("")
    lines.append("## Sample archive members (first 20)")
    for archive in archives[:3]:
        lines.append(f"- Archive: {archive}")
        info = inspected.get(archive, {})
        if info.get("error"):
            lines.append(f"  - ERROR: {info['error']}")
            continue
        for member in info.get("members", []):
            lines.append(f"  - {member}")
        if info.get("csv_members"):
            lines.append("  - CSV candidates:")
            for member in info.get("csv_members", []):
                lines.append(f"    - {member}")
    lines.append("")
    lines.append("## Extraction targets")
    if extracted_dirs:
        for dest in extracted_dirs:
            lines.append(f"- {dest}")
    else:
        lines.append("- none")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _fhrma_hex_preview(paths: list[Path], limit: int = 5) -> list[dict]:
    previews: list[dict] = []
    for path in paths[:limit]:
        try:
            with open(path, "rb") as f:
                data = f.read(64)
            previews.append({
                "path": str(path),
                "hex": data.hex(),
            })
        except Exception as exc:
            previews.append({
                "path": str(path),
                "error": str(exc),
            })
    return previews


def _write_fhrma_binary_report(
    report_path: Path,
    root: Path | None,
    binary_files: list[Path],
    decode_stats: dict,
) -> None:
    lines = ["# FHRMA Binary Decoding Report (V6.2)", ""]
    if root is None or not root.exists():
        lines.append("Dataset root not found.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(f"Resolved root: {root}")
    lines.append("")

    if not binary_files:
        lines.append("No binary files (.fhr/.fhrm) found.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    sizes = np.array([p.stat().st_size for p in binary_files], dtype=float)
    ext_counts: Dict[str, int] = {}
    header_counts: Dict[str, int] = {}
    for path in binary_files:
        ext = path.suffix.lower() or "<none>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    for path in binary_files[:1000]:
        try:
            with open(path, "rb") as f:
                header = f.read(4).hex()
            header_counts[header] = header_counts.get(header, 0) + 1
        except Exception:
            continue
    lines.append(f"Binary files: {len(binary_files)}")
    lines.append(f"Size (bytes) min/median/max: {int(sizes.min())} / {int(np.median(sizes))} / {int(sizes.max())}")
    lines.append("")
    lines.append("## Counts by extension")
    for ext, count in sorted(ext_counts.items(), key=lambda x: x[0]):
        lines.append(f"- {ext}: {count}")
    lines.append("")
    lines.append("## Header byte frequency (first 4 bytes, sample up to 1000 files)")
    for header, count in sorted(header_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"- {header}: {count}")
    lines.append("")
    lines.append("## Sample filenames (up to 20)")
    for path in binary_files[:20]:
        lines.append(f"- {path}")

    lines.append("")
    lines.append("## Hex preview (first 64 bytes, up to 5 files)")
    for item in decode_stats.get("hex_previews", []):
        lines.append(f"- {item.get('path')}")
        if item.get("error"):
            lines.append(f"  - ERROR: {item['error']}")
        else:
            lines.append(f"  - {item.get('hex')}")

    lines.append("")
    lines.append("## Decode summary")
    lines.append(f"- successes: {decode_stats.get('success', 0)}")
    lines.append(f"- failures: {decode_stats.get('failure', 0)}")
    lines.append("### Methods used")
    for method, count in sorted(decode_stats.get("methods", {}).items()):
        lines.append(f"- {method}: {count}")
    lines.append("### Skip reasons")
    for reason, count in sorted(decode_stats.get("skips", {}).items()):
        lines.append(f"- {reason}: {count}")
    lines.append("### fs_inferred")
    for flag, count in sorted(decode_stats.get("fs_inferred", {}).items()):
        lines.append(f"- {flag}: {count}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V6 training pack (Pre-AI only).")
    parser.add_argument("--data-root", default="data", help="Root folder for CTU-CHB/CTGDL/FHRMA datasets")
    parser.add_argument("--out", default="training_pack_v6_2.zip", help="Output zip path")
    parser.add_argument("--log-dir", default=None, help="Override log directory")
    args = parser.parse_args()

    log_dir = _ensure_log_dir(Path(args.log_dir) if args.log_dir else None)
    log_path = _configure_logging(log_dir)

    cfg = load_runtime_config()
    policy = load_quality_policy()
    apply_strict_warnings(cfg.strict_mode)

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting V6 training pack build")
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info("Data root: %s", data_root)
    LOGGER.info("Output zip: %s", out_path)

    datasets = {
        "CTU-CHB": _find_dataset_root(data_root, ["CTU-CHB", "CTU_CHB", "ctu-chb", "ctu"]),
        "CTGDL": _find_dataset_root(data_root, ["CTGDL", "ctgdl", "ctgd"]),
        "FHRMA": _find_dataset_root(data_root, ["FHRMA", "fhrma", "FSdataset", "fsdataset", "fhr-ma"]),
    }

    fhrma_report = Path("REPORTS") / "FHRMA_FORENSICS_2026.txt"
    _write_fhrma_forensics(fhrma_report, datasets["FHRMA"])
    ctgdl_suggestion_entries: list[dict] = []

    ctgdl_archives: list[Path] = []
    ctgdl_inspected: dict[Path, dict] = {}
    ctgdl_extracted_dirs: list[Path] = []
    ctgdl_scan_roots: list[Path] = []
    fhrma_binary_files: list[Path] = []
    fhrma_decode_stats: dict = {
        "success": 0,
        "failure": 0,
        "methods": {},
        "skips": {},
        "fs_inferred": {},
        "hex_previews": [],
    }

    if datasets["CTGDL"] is not None:
        ctgdl_root = datasets["CTGDL"]
        ctgdl_archives = sorted(list(ctgdl_root.rglob("*.tar.gz")) + list(ctgdl_root.rglob("*.tgz")))
        for archive in ctgdl_archives:
            ctgdl_inspected[archive] = _inspect_tar_members(archive)
        extract_root = data_root / "_extracted" / "ctgdl"
        extract_root.mkdir(parents=True, exist_ok=True)
        for archive in ctgdl_archives:
            dest, extracted = _safe_extract_tar(archive, extract_root)
            if extracted:
                ctgdl_extracted_dirs.append(dest)
            else:
                ctgdl_extracted_dirs.append(dest)
        ctgdl_scan_roots = [ctgdl_root] + ctgdl_extracted_dirs

        ctgdl_report = Path("REPORTS") / "CTGDL_EXTRACTION_REPORT_2026.txt"
        _write_ctgdl_extraction_report(ctgdl_report, ctgdl_archives, ctgdl_inspected, ctgdl_extracted_dirs)

    if datasets["FHRMA"] is not None and datasets["FHRMA"].exists():
        fhrma_root = datasets["FHRMA"]
        fhrma_binary_files = sorted(
            set(
                list(fhrma_root.rglob("*.fhr"))
                + list(fhrma_root.rglob("*.fhrm"))
                + list(fhrma_root.rglob("*.rcf"))
                + list(fhrma_root.rglob("*.rcfm"))
                + list(fhrma_root.rglob("*.dat"))
            )
        )
        fhrma_decode_stats["hex_previews"] = _fhrma_hex_preview(fhrma_binary_files)

    manifest = {
        "build_timestamp": datetime.now().isoformat(),
        "git_commit": _get_git_commit(),
        "runtime_config": {
            "fs_hz": cfg.fs_hz,
            "window_minutes": cfg.window_minutes,
            "stride_minutes": cfg.stride_minutes,
            "min_window_minutes": cfg.min_window_minutes,
        },
        "quality_policy": policy,
        "datasets": {},
        "patients": [],
        "ctgdl_suggestions": {
            "files_with_suggestions": 0,
            "top_columns": {},
        },
        "stats": {
            "counts": {
                "total_records": 0,
                "included": 0,
                "skipped": 0,
            },
            "counts_by_dataset": {},
            "counts_by_target_task": {},
            "skipped_by_reason": {},
            "duration_minutes": {"min": 0.0, "mean": 0.0, "max": 0.0},
            "quality_class_counts": {},
        },
    }

    records_dir = log_dir / "training_pack_v6" / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    durations = []

    def _skip(reason: str) -> None:
        stats = manifest["stats"]
        stats["counts"]["skipped"] += 1
        stats["skipped_by_reason"][reason] = stats["skipped_by_reason"].get(reason, 0) + 1

    def _record_patient(entry: Dict) -> None:
        manifest["patients"].append(entry)
        manifest["stats"]["counts"]["total_records"] += 1
        ds = entry.get("source", "UNKNOWN")
        counts = manifest["stats"]["counts_by_dataset"]
        counts[ds] = counts.get(ds, {"total": 0, "included": 0, "skipped": 0})
        counts[ds]["total"] += 1
        if entry.get("included"):
            counts[ds]["included"] += 1
        else:
            counts[ds]["skipped"] += 1
        task = entry.get("target_task", "Unknown")
        tasks = manifest["stats"]["counts_by_target_task"]
        tasks[task] = tasks.get(task, 0) + 1

    def _quality_count(records: List[Dict]) -> None:
        stats = manifest["stats"]["quality_class_counts"]
        for rec in records:
            key = rec.get("quality_class", "UNKNOWN")
            stats[key] = stats.get(key, 0) + 1

    def _quality_distribution(records: List[Dict]) -> Dict[str, float]:
        total = len(records)
        if total == 0:
            return {"HIGH": 0.0, "MED": 0.0, "LOW": 0.0}
        counts = {"HIGH": 0, "MED": 0, "LOW": 0}
        for rec in records:
            key = rec.get("quality_class", "LOW")
            if key not in counts:
                key = "LOW"
            counts[key] += 1
        return {k: counts[k] / total for k in counts}

    def _label_type(labels: Dict[str, object]) -> str:
        if not labels:
            return "none"
        keys = {str(k).lower() for k in labels.keys()}
        if {"ph", "outcome"}.intersection(keys):
            return "outcome"
        if {"quality_label", "noise", "annotation", "label", "class"}.intersection(keys):
            return "quality_label"
        return "annotation"

    def _target_task(source: str, labels: Dict[str, object]) -> str:
        if source == "FHRMA":
            return "Quality"
        if source == "CTU-CHB":
            return "Outcome"
        if source == "CTGDL":
            keys = {str(k).lower() for k in labels.keys()} if labels else set()
            if {"ph", "outcome"}.intersection(keys):
                return "Outcome"
            return "Anatomy"
        return "Unknown"

    for name, root in datasets.items():
        manifest["datasets"][name] = {
            "path": str(root) if root else None,
            "found": root is not None,
        }
        if name == "CTGDL":
            manifest["datasets"][name]["archives_found"] = len(ctgdl_archives)
            manifest["datasets"][name]["archives_extracted"] = len(ctgdl_extracted_dirs)
            manifest["datasets"][name]["extracted_dirs"] = [str(p) for p in ctgdl_extracted_dirs]
        if root is None:
            LOGGER.info("Dataset %s not found (skipping)", name)
            continue

        if name == "CTU-CHB":
            records = discover_ctu_records(root)
            LOGGER.info("CTU-CHB records found: %d", len(records))
            for hea_path in records:
                result = load_ctu_record(hea_path)
                patient_id = hea_path.stem
                if result.record is None:
                    reason = result.skip_reason or "LOAD_FAIL"
                    LOGGER.info("CTU-CHB %s skipped: %s", patient_id, reason)
                    _record_patient({
                        "patient_id": patient_id,
                        "source": name,
                        "target_task": "Outcome",
                        "included": False,
                        "skip_reason": reason,
                    })
                    _skip(reason)
                    continue
                record = result.record
                reject_detail = record_reject_reason(record.record_quality, policy)
                target_task = _target_task(record.source, record.labels)
                record.meta["target_task"] = target_task
                entry = {
                    "patient_id": record.patient_id,
                    "source": record.source,
                    "duration_minutes": record.record_quality.get("duration_minutes", 0.0),
                    "record_quality": record.record_quality,
                    "labels_present": bool(record.labels),
                    "label_type": _label_type(record.labels),
                    "target_task": target_task,
                    "included": False,
                    "skip_reason": None,
                    "reject_detail": reject_detail,
                    "short_case": False,
                    "avg_quality": None,
                    "gap_fill": record.meta.get("gap_fill"),
                }
                if reject_detail:
                    entry["skip_reason"] = "RECORD_REJECT_POLICY"
                elif record.fs_hz != cfg.fs_hz:
                    entry["skip_reason"] = "FS_MISMATCH"
                elif len(record.fhr_raw) != len(record.uc_raw):
                    entry["skip_reason"] = "ALIGNMENT_FAIL"
                else:
                    try:
                        assert_raw_invariants(
                            record.fhr_raw,
                            record.uc_raw,
                            fs_hz=cfg.fs_hz,
                            min_case_minutes=cfg.min_case_minutes,
                            min_window_minutes=cfg.min_window_minutes,
                            tag=f"RAW:{record.patient_id}",
                        )
                        window_records = run_pre_ai(
                            record.patient_id,
                            record.fhr_raw,
                            record.uc_raw,
                        )
                        entry["avg_quality"] = _quality_distribution(window_records)
                        _quality_count(window_records)
                    except WarmupError:
                        entry["skip_reason"] = "WARMUP_ERROR"
                    except Exception as exc:
                        entry["skip_reason"] = f"INVARIANT_FAIL: {exc}"
                    else:
                        duration = entry["duration_minutes"]
                        entry["short_case"] = cfg.min_case_minutes <= duration < cfg.recommended_case_minutes
                        _save_npz(records_dir / f"{record.patient_id}.npz", record)
                        entry["included"] = True
                        manifest["stats"]["counts"]["included"] += 1
                        durations.append(duration)
                if entry["skip_reason"]:
                    _skip(entry["skip_reason"])
                _record_patient(entry)

        if name == "CTGDL":
            scan_roots = ctgdl_scan_roots or [root]
            csv_files = []
            for scan_root in scan_roots:
                if scan_root is None or not scan_root.exists():
                    continue
                csv_files.extend(scan_root.rglob("*.csv"))
            csv_files = sorted({p for p in csv_files})
            LOGGER.info("CTGDL records found: %d", len(csv_files))
            for csv_path in csv_files:
                result = load_ctgdl_record(csv_path)
                patient_id = csv_path.stem
                if result.record is None:
                    reason = result.skip_reason or "LOAD_FAIL"
                    if reason == "FHR_COLUMN_MISSING" and result.suggestions:
                        ctgdl_suggestion_entries.append(result.suggestions)
                        manifest["ctgdl_suggestions"]["files_with_suggestions"] += 1
                        for cand in result.suggestions.get("top_candidates", []):
                            name = cand.get("column")
                            if name:
                                manifest["ctgdl_suggestions"]["top_columns"][name] = (
                                    manifest["ctgdl_suggestions"]["top_columns"].get(name, 0) + 1
                                )
                    LOGGER.info("CTGDL %s skipped: %s", patient_id, reason)
                    _record_patient({
                        "patient_id": patient_id,
                        "source": name,
                        "target_task": "Anatomy",
                        "included": False,
                        "skip_reason": reason,
                    })
                    _skip(reason)
                    continue
                record = result.record
                reject_detail = record_reject_reason(record.record_quality, policy)
                target_task = _target_task(record.source, record.labels)
                record.meta["target_task"] = target_task
                entry = {
                    "patient_id": record.patient_id,
                    "source": record.source,
                    "duration_minutes": record.record_quality.get("duration_minutes", 0.0),
                    "record_quality": record.record_quality,
                    "labels_present": bool(record.labels),
                    "label_type": _label_type(record.labels),
                    "target_task": target_task,
                    "included": False,
                    "skip_reason": None,
                    "reject_detail": reject_detail,
                    "short_case": False,
                    "avg_quality": None,
                    "gap_fill": record.meta.get("gap_fill"),
                }
                if reject_detail:
                    entry["skip_reason"] = "RECORD_REJECT_POLICY"
                elif record.fs_hz != cfg.fs_hz:
                    entry["skip_reason"] = "FS_MISMATCH"
                elif len(record.fhr_raw) != len(record.uc_raw):
                    entry["skip_reason"] = "ALIGNMENT_FAIL"
                else:
                    try:
                        assert_raw_invariants(
                            record.fhr_raw,
                            record.uc_raw,
                            fs_hz=cfg.fs_hz,
                            min_case_minutes=cfg.min_case_minutes,
                            min_window_minutes=cfg.min_window_minutes,
                            tag=f"RAW:{record.patient_id}",
                        )
                        window_records = run_pre_ai(
                            record.patient_id,
                            record.fhr_raw,
                            record.uc_raw,
                        )
                        entry["avg_quality"] = _quality_distribution(window_records)
                        _quality_count(window_records)
                    except WarmupError:
                        entry["skip_reason"] = "WARMUP_ERROR"
                    except Exception as exc:
                        entry["skip_reason"] = f"INVARIANT_FAIL: {exc}"
                    else:
                        duration = entry["duration_minutes"]
                        entry["short_case"] = cfg.min_case_minutes <= duration < cfg.recommended_case_minutes
                        _save_npz(records_dir / f"{record.patient_id}.npz", record)
                        entry["included"] = True
                        manifest["stats"]["counts"]["included"] += 1
                        durations.append(duration)
                if entry["skip_reason"]:
                    _skip(entry["skip_reason"])
                _record_patient(entry)

        if name == "FHRMA":
            csv_files = discover_fhrma_records(root)
            LOGGER.info("FHRMA records found: %d", len(csv_files))
            for csv_path in csv_files:
                result = load_fhrma_record(csv_path)
                patient_id = csv_path.stem
                if result.record is None:
                    reason = result.skip_reason or "LOAD_FAIL"
                    fhrma_decode_stats["failure"] += 1
                    fhrma_decode_stats["skips"][reason] = fhrma_decode_stats["skips"].get(reason, 0) + 1
                    if reason == "MAT_KEYS_UNKNOWN" and result.details:
                        with (fhrma_report).open("a", encoding="utf-8") as f:
                            f.write("\n## MAT_KEYS_UNKNOWN\n")
                            f.write(f"- File: {csv_path}\n")
                            f.write(f"- Keys: {result.details.get('keys')}\n")
                    LOGGER.info("FHRMA %s skipped: %s", patient_id, reason)
                    _record_patient({
                        "patient_id": patient_id,
                        "source": name,
                        "target_task": "Quality",
                        "included": False,
                        "skip_reason": reason,
                    })
                    _skip(reason)
                    continue
                record = result.record
                if result.details:
                    method = result.details.get("decode_method", "unknown")
                    fhrma_decode_stats["methods"][method] = fhrma_decode_stats["methods"].get(method, 0) + 1
                    inferred = str(bool(result.details.get("fs_inferred", False)))
                    fhrma_decode_stats["fs_inferred"][inferred] = (
                        fhrma_decode_stats["fs_inferred"].get(inferred, 0) + 1
                    )
                fhrma_decode_stats["success"] += 1
                reject_detail = record_reject_reason(record.record_quality, policy)
                target_task = _target_task(record.source, record.labels)
                record.meta["target_task"] = target_task
                entry = {
                    "patient_id": record.patient_id,
                    "source": record.source,
                    "duration_minutes": record.record_quality.get("duration_minutes", 0.0),
                    "record_quality": record.record_quality,
                    "labels_present": bool(record.labels),
                    "label_type": _label_type(record.labels),
                    "target_task": target_task,
                    "included": False,
                    "skip_reason": None,
                    "reject_detail": reject_detail,
                    "short_case": False,
                    "avg_quality": None,
                    "gap_fill": record.meta.get("gap_fill"),
                }
                if record.fs_hz != cfg.fs_hz:
                    entry["skip_reason"] = "FS_MISMATCH"
                elif len(record.fhr_raw) != len(record.uc_raw):
                    entry["skip_reason"] = "ALIGNMENT_FAIL"
                else:
                    try:
                        assert_raw_invariants(
                            record.fhr_raw,
                            record.uc_raw,
                            fs_hz=cfg.fs_hz,
                            min_case_minutes=cfg.min_case_minutes,
                            min_window_minutes=cfg.min_window_minutes,
                            tag=f"RAW:{record.patient_id}",
                        )
                        window_records = run_pre_ai(
                            record.patient_id,
                            record.fhr_raw,
                            record.uc_raw,
                        )
                        entry["avg_quality"] = _quality_distribution(window_records)
                        _quality_count(window_records)
                    except WarmupError:
                        entry["skip_reason"] = "WARMUP_ERROR"
                    except Exception as exc:
                        entry["skip_reason"] = f"INVARIANT_FAIL: {exc}"
                    else:
                        duration = entry["duration_minutes"]
                        entry["short_case"] = cfg.min_case_minutes <= duration < cfg.recommended_case_minutes
                        _save_npz(records_dir / f"{record.patient_id}.npz", record)
                        entry["included"] = True
                        manifest["stats"]["counts"]["included"] += 1
                        durations.append(duration)
                if entry["skip_reason"]:
                    _skip(entry["skip_reason"])
                _record_patient(entry)

    fhrma_binary_report = Path("REPORTS") / "FHRMA_BINARY_DECODING_REPORT_2026.txt"
    _write_fhrma_binary_report(fhrma_binary_report, datasets.get("FHRMA"), fhrma_binary_files, fhrma_decode_stats)
    if "FHRMA" in manifest["datasets"]:
        manifest["datasets"]["FHRMA"]["decode_methods"] = fhrma_decode_stats.get("methods", {})
        manifest["datasets"]["FHRMA"]["decode_success"] = fhrma_decode_stats.get("success", 0)
        manifest["datasets"]["FHRMA"]["decode_failure"] = fhrma_decode_stats.get("failure", 0)

    if durations:
        manifest["stats"]["duration_minutes"] = {
            "min": float(np.min(durations)),
            "mean": float(np.mean(durations)),
            "max": float(np.max(durations)),
        }

    ctgdl_report = Path("REPORTS") / "CTGDL_COLUMN_SUGGESTIONS_2026.txt"
    _write_ctgdl_suggestions(ctgdl_report, ctgdl_suggestion_entries)

    manifest_path = log_dir / "training_pack_v6" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.json")
        for npz_path in sorted(records_dir.glob("*.npz")):
            zf.write(npz_path, f"records/{npz_path.name}")

    report_path = Path("REPORTS") / "DATASET_SUMMARY_V6_2.txt"
    _write_summary(manifest, report_path)

    LOGGER.info("Manifest written to %s", manifest_path)
    LOGGER.info("Dataset summary written to %s", report_path)
    LOGGER.info("Training pack zip written to %s", out_path)
    LOGGER.info("Records included: %d", manifest["stats"]["counts"]["included"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
