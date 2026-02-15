"""
Stage 2: Windowing + Quality Gate Pipeline

This module implements the windowing stage of the SentinelFetal pipeline.
It converts Stage 1 artifacts into a canonical window index with quality metrics.

Key Design Decision (from NotebookLM consultation):
- The core window processing logic (`process_single_window`) is designed to be
  reusable for both BATCH (training) and STREAMING (real-time) contexts.
- In BATCH mode: This script iterates over all historical data.
- In STREAMING mode: The same `process_single_window` function can be called
  by a Ring Buffer orchestrator in real-time.

Inputs (from Stage 1):
    - X_norm_4hz.npy: Normalized data matrix (N, Max_T, 2)
    - mask_fhr.npy: FHR invalidity mask (N, Max_T)
    - mask_uc.npy: UC invalidity mask (N, Max_T)
    - manifest.csv: Record metadata including raw_len

Outputs:
    - windows_index.parquet: Window index with quality metrics
    - stage2_qc_report.json: Global QC statistics
    - STAGE2_TRUTH.md: Truth document

Author: SentinelFetal Team
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class WindowQualityThresholds:
    """Thresholds for window quality classification.
    
    Quality Classes:
        GOOD: inv_rate < inv_rate_good (< 5%)
        MEDIUM: inv_rate_good <= inv_rate < inv_rate_medium (5-15%)
        LOW: inv_rate_medium <= inv_rate < inv_rate_low (15-30%)
        FAIL: inv_rate >= inv_rate_low (>= 30%)
    """
    inv_rate_good: float = 0.05      # < 5% invalid → GOOD
    inv_rate_medium: float = 0.15    # 5-15% → MEDIUM
    inv_rate_low: float = 0.30       # 15-30% → LOW
    # > 30% → FAIL


@dataclass
class WindowingConfig:
    """Configuration for the windowing pipeline.
    
    Attributes:
        fs: Sampling frequency in Hz (must be 4 for CTU-CHB)
        window_minutes: Window length in minutes
        stride_minutes: Stride between windows in minutes
        thresholds: Quality classification thresholds
    """
    fs: int = 4
    window_minutes: int = 20
    stride_minutes: int = 5
    thresholds: WindowQualityThresholds = field(default_factory=WindowQualityThresholds)
    
    @property
    def window_samples(self) -> int:
        """Window size in samples."""
        return self.window_minutes * 60 * self.fs  # 20 * 60 * 4 = 4800
    
    @property
    def stride_samples(self) -> int:
        """Stride size in samples."""
        return self.stride_minutes * 60 * self.fs  # 5 * 60 * 4 = 1200


# =============================================================================
# Core Window Processing (SHARED for Batch & Real-Time)
# =============================================================================

@dataclass
class WindowQualityResult:
    """Result of window quality assessment.
    
    This dataclass is the output of process_single_window and can be used
    by both batch and real-time pipelines.
    """
    inv_rate_fhr: float
    inv_rate_uc: float
    quality_class: str  # GOOD, MEDIUM, LOW, FAIL
    valid_for_ai: bool
    valid_for_rules: bool
    touches_padding: bool


def process_single_window(
    mask_fhr_window: np.ndarray,
    mask_uc_window: np.ndarray,
    thresholds: WindowQualityThresholds,
    touches_padding: bool = False
) -> WindowQualityResult:
    """
    Process a single window and compute quality metrics.
    
    THIS IS THE CORE SHARED FUNCTION.
    - In BATCH mode: Called in a loop over all windows.
    - In REAL-TIME mode: Called once per trigger (e.g., every 5 minutes).
    
    Args:
        mask_fhr_window: Boolean mask for FHR (True = invalid)
        mask_uc_window: Boolean mask for UC (True = invalid)
        thresholds: Quality thresholds for classification
        touches_padding: Whether this window extends into padding area
        
    Returns:
        WindowQualityResult with all quality metrics
    """
    # Compute invalidity rates
    inv_rate_fhr = float(mask_fhr_window.mean())
    inv_rate_uc = float(mask_uc_window.mean())
    
    # Take the worse of the two channels
    inv_rate_max = max(inv_rate_fhr, inv_rate_uc)
    
    # Classify quality
    if touches_padding:
        quality_class = "FAIL"
    elif inv_rate_max < thresholds.inv_rate_good:
        quality_class = "GOOD"
    elif inv_rate_max < thresholds.inv_rate_medium:
        quality_class = "MEDIUM"
    elif inv_rate_max < thresholds.inv_rate_low:
        quality_class = "LOW"
    else:
        quality_class = "FAIL"
    
    # Determine validity for different use cases
    # AI models require GOOD or MEDIUM quality
    valid_for_ai = quality_class in ("GOOD", "MEDIUM")
    # Rules can work with LOW quality (degraded mode)
    valid_for_rules = quality_class in ("GOOD", "MEDIUM", "LOW")
    
    return WindowQualityResult(
        inv_rate_fhr=inv_rate_fhr,
        inv_rate_uc=inv_rate_uc,
        quality_class=quality_class,
        valid_for_ai=valid_for_ai,
        valid_for_rules=valid_for_rules,
        touches_padding=touches_padding
    )


# =============================================================================
# Batch Pipeline Functions
# =============================================================================

def generate_windows_for_record(
    record_idx: int,
    record_id: str,
    raw_len: int,
    mask_fhr: np.ndarray,
    mask_uc: np.ndarray,
    config: WindowingConfig
) -> List[Dict[str, Any]]:
    """
    Generate all windows for a single record.
    
    Implements the batch-mode windowing logic:
    - Iterate from start=0 to raw_len - W with stride S
    - Skip windows that extend into padding (beyond raw_len)
    
    Args:
        record_idx: Index in the data array
        record_id: Record identifier
        raw_len: Actual length of the record (before padding)
        mask_fhr: Full FHR mask for this record
        mask_uc: Full UC mask for this record
        config: Windowing configuration
        
    Returns:
        List of window dictionaries
    """
    W = config.window_samples
    S = config.stride_samples
    
    windows = []
    window_idx = 0
    
    # Iterate with stride, ensuring window stays within raw_len
    for start in range(0, raw_len - W + 1, S):
        end = start + W
        
        # Extract window masks
        mask_fhr_win = mask_fhr[start:end]
        mask_uc_win = mask_uc[start:end]
        
        # Check if window touches padding (shouldn't happen with this loop)
        touches_padding = end > raw_len
        
        # Process window
        result = process_single_window(
            mask_fhr_win, mask_uc_win,
            config.thresholds,
            touches_padding
        )
        
        # Build window record
        window_record = {
            "record_id": record_id,
            "array_idx": record_idx,
            "window_idx": window_idx,
            "start": start,
            "end": end,
            "start_time_sec": start / config.fs,
            "end_time_sec": end / config.fs,
            "inv_rate_fhr": result.inv_rate_fhr,
            "inv_rate_uc": result.inv_rate_uc,
            "quality_class": result.quality_class,
            "valid_for_ai": result.valid_for_ai,
            "valid_for_rules": result.valid_for_rules,
            "touches_padding": result.touches_padding
        }
        
        windows.append(window_record)
        window_idx += 1
    
    return windows


def build_windows_index(
    input_dir: Path,
    config: WindowingConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build the complete windows index from Stage 1 artifacts.
    
    Args:
        input_dir: Directory containing Stage 1 outputs
        config: Windowing configuration
        
    Returns:
        Tuple of (windows_df, qc_stats)
    """
    logger.info(f"Loading Stage 1 artifacts from {input_dir}")
    
    # Load artifacts
    X = np.load(input_dir / "X_norm_4hz.npy", mmap_mode='r')
    mask_fhr = np.load(input_dir / "mask_fhr.npy", mmap_mode='r')
    mask_uc = np.load(input_dir / "mask_uc.npy", mmap_mode='r')
    manifest = pd.read_csv(input_dir / "manifest.csv")
    
    logger.info(f"Loaded {len(manifest)} records, shape: {X.shape}")
    logger.info(f"Window config: {config.window_minutes}min window, {config.stride_minutes}min stride")
    
    # Initialize QC stats
    qc_stats = {
        "config": asdict(config),
        "total_records": len(manifest),
        "records_with_windows": 0,
        "records_too_short": 0,
        "total_windows": 0,
        "windows_by_quality": {"GOOD": 0, "MEDIUM": 0, "LOW": 0, "FAIL": 0},
        "windows_valid_for_ai": 0,
        "windows_valid_for_rules": 0,
        "avg_windows_per_record": 0.0,
        "timestamp": datetime.now().isoformat()
    }
    
    all_windows = []
    
    # Process each record
    for _, row in manifest.iterrows():
        record_id = row["record_id"]
        array_idx = int(row["array_idx"])
        raw_len = int(row["samples"])
        
        # Check if record is long enough for at least one window
        if raw_len < config.window_samples:
            qc_stats["records_too_short"] += 1
            logger.debug(f"Record {record_id} too short ({raw_len} < {config.window_samples})")
            continue
        
        # Generate windows
        windows = generate_windows_for_record(
            record_idx=array_idx,
            record_id=record_id,
            raw_len=raw_len,
            mask_fhr=mask_fhr[array_idx],
            mask_uc=mask_uc[array_idx],
            config=config
        )
        
        if windows:
            qc_stats["records_with_windows"] += 1
            all_windows.extend(windows)
    
    # Create DataFrame
    windows_df = pd.DataFrame(all_windows)
    
    # Update QC stats
    qc_stats["total_windows"] = len(windows_df)
    
    if len(windows_df) > 0:
        qc_stats["windows_by_quality"] = windows_df["quality_class"].value_counts().to_dict()
        qc_stats["windows_valid_for_ai"] = int(windows_df["valid_for_ai"].sum())
        qc_stats["windows_valid_for_rules"] = int(windows_df["valid_for_rules"].sum())
        qc_stats["avg_windows_per_record"] = len(windows_df) / qc_stats["records_with_windows"]
    
    logger.info(f"Generated {len(windows_df)} windows from {qc_stats['records_with_windows']} records")
    logger.info(f"Quality distribution: {qc_stats['windows_by_quality']}")
    
    return windows_df, qc_stats


def save_artifacts(
    windows_df: pd.DataFrame,
    qc_stats: Dict[str, Any],
    output_dir: Path
) -> None:
    """Save Stage 2 artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save windows index - try parquet first, fall back to CSV
    windows_path_parquet = output_dir / "windows_index.parquet"
    windows_path_csv = output_dir / "windows_index.csv"
    
    try:
        windows_df.to_parquet(windows_path_parquet, index=False)
        logger.info(f"Saved windows index to {windows_path_parquet}")
    except ImportError:
        # Fallback to CSV if pyarrow/fastparquet not available
        windows_df.to_csv(windows_path_csv, index=False)
        logger.info(f"Saved windows index to {windows_path_csv} (parquet unavailable)")
    
    # Save QC report
    qc_path = output_dir / "stage2_qc_report.json"
    with open(qc_path, "w") as f:
        json.dump(qc_stats, f, indent=2, default=str)
    logger.info(f"Saved QC report to {qc_path}")


def generate_truth_md(
    qc_stats: Dict[str, Any],
    output_dir: Path
) -> Path:
    """Generate STAGE2_TRUTH.md document."""
    
    config = qc_stats.get("config", {})
    by_quality = qc_stats.get("windows_by_quality", {})
    
    md = f"""<div dir="rtl" style="text-align: right;">

# מסמך אמת (Truth Source) - Stage 2
**נכון לתאריך:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 1. תקציר מנהלים (Purpose)
שלב 2 (Stage 2) אחראי על חיתוך הנתונים לחלונות בגודל קבוע ומדרג את איכות כל חלון.
הוא מייצר את **אינדקס החלונות** שמשמש את כל השלבים הבאים (אימון, חוקים, Real-Time).

## 2. תמונת מצב נוכחית (Current State)

### קונפיגורציה
*   **תדר דגימה:** {config.get('fs', 4)}Hz
*   **אורך חלון:** {config.get('window_minutes', 20)} דקות ({config.get('window_minutes', 20) * 60 * config.get('fs', 4)} דגימות)
*   **קפיצה (Stride):** {config.get('stride_minutes', 5)} דקות

### סטטיסטיקות
*   **סה"כ רשומות:** {qc_stats.get('total_records', 0)}
*   **רשומות עם חלונות:** {qc_stats.get('records_with_windows', 0)}
*   **רשומות קצרות מדי:** {qc_stats.get('records_too_short', 0)}
*   **סה"כ חלונות:** {qc_stats.get('total_windows', 0):,}
*   **ממוצע חלונות לרשומה:** {qc_stats.get('avg_windows_per_record', 0):.1f}

### התפלגות איכות
| Quality Class | כמות | אחוז |
|---------------|------|------|
| GOOD | {by_quality.get('GOOD', 0):,} | {by_quality.get('GOOD', 0) / max(qc_stats.get('total_windows', 1), 1) * 100:.1f}% |
| MEDIUM | {by_quality.get('MEDIUM', 0):,} | {by_quality.get('MEDIUM', 0) / max(qc_stats.get('total_windows', 1), 1) * 100:.1f}% |
| LOW | {by_quality.get('LOW', 0):,} | {by_quality.get('LOW', 0) / max(qc_stats.get('total_windows', 1), 1) * 100:.1f}% |
| FAIL | {by_quality.get('FAIL', 0):,} | {by_quality.get('FAIL', 0) / max(qc_stats.get('total_windows', 1), 1) * 100:.1f}% |

### כשרות לשימוש
*   **כשרים למודלים (AI):** {qc_stats.get('windows_valid_for_ai', 0):,}
*   **כשרים לחוקים:** {qc_stats.get('windows_valid_for_rules', 0):,}

## 3. ארטיפקטים (Artifacts)
*   `windows_index.parquet` - אינדקס החלונות
*   `stage2_qc_report.json` - דוח איכות

## 4. שימוש חוזר (Reusability)
פונקציית הליבה `process_single_window` מיועדת לשימוש חוזר:
*   **באימון (Batch):** נקראת בלולאה על כל ההיסטוריה
*   **בזמן אמת (Streaming):** נקראת פעם ב-5 דקות על ה-Ring Buffer

</div>
"""
    
    truth_path = output_dir / "STAGE2_TRUTH.md"
    with open(truth_path, "w") as f:
        f.write(md)
    
    logger.info(f"Saved truth document to {truth_path}")
    return truth_path


# =============================================================================
# Main Pipeline Entry Point
# =============================================================================

def run_stage2_pipeline(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    config: Optional[WindowingConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the complete Stage 2 pipeline.
    
    Args:
        input_dir: Directory containing Stage 1 outputs
        output_dir: Directory for Stage 2 outputs (defaults to input_dir)
        config: Windowing configuration (uses defaults if None)
        
    Returns:
        Tuple of (windows_df, qc_stats)
    """
    if output_dir is None:
        output_dir = input_dir
    
    if config is None:
        config = WindowingConfig()
    
    logger.info("=" * 60)
    logger.info("Stage 2: Windowing + Quality Gate Pipeline")
    logger.info("=" * 60)
    
    # Build windows index
    windows_df, qc_stats = build_windows_index(input_dir, config)
    
    # Save artifacts
    save_artifacts(windows_df, qc_stats, output_dir)
    
    # Generate truth document
    generate_truth_md(qc_stats, output_dir)
    
    logger.info("=" * 60)
    logger.info("Stage 2 Pipeline Complete")
    logger.info("=" * 60)
    
    return windows_df, qc_stats
