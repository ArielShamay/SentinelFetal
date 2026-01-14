"""
Dataset Preparation Script for SentinelFetal.

This script runs the full pipeline on the CTU-UHB dataset:
1. Load CTG records
2. Preprocess signals
3. Extract rule-based features
4. Extract MOMENT embeddings (REAL model, not mock)
5. Build 1035-dim feature vectors
6. Save X.npy and y.npy for classifier training

Usage:
    python src/training/prepare_data.py --limit 5  # Test with 5 records
    python src/training/prepare_data.py            # Process all records

CRITICAL: This script uses use_mock=False to ensure REAL MOMENT embeddings!
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.data.loader import CTUDataLoader, DataLoaderError
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules.baseline import calculate_baseline
from src.rules.variability import calculate_variability
from src.rules.decelerations import detect_decelerations
from src.rules.tachysystole import detect_tachysystole
from src.rules.sinusoidal import detect_sinusoidal_pattern
from src.models.moment_encoder import MomentFeatureExtractor, extract_embeddings_sliding_window
from src.models.fusion import build_feature_vector, build_feature_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    data_dir: str = "data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0"
    output_dir: str = "data/processed"
    limit: Optional[int] = None
    use_mock: bool = False  # CRITICAL: Set to False for real MOMENT embeddings!
    device: str = "auto"
    window_minutes: float = 10.0
    step_minutes: float = 1.0
    sampling_rate: float = 4.0


def run_rule_engine(fhr: np.ndarray, uc: np.ndarray, sampling_rate: float = 4.0) -> dict:
    """
    Run all rule-based analyzers on a CTG signal.
    
    Args:
        fhr: Preprocessed FHR signal.
        uc: Preprocessed UC signal.
        sampling_rate: Signal sampling rate in Hz.
        
    Returns:
        Dictionary with all rule engine results.
    """
    # Baseline analysis
    baseline_result = calculate_baseline(fhr, sampling_rate=sampling_rate)
    
    # Variability analysis
    variability_result = calculate_variability(fhr, sampling_rate=sampling_rate)
    
    # Deceleration detection
    decelerations = detect_decelerations(fhr, uc, baseline_result.value, sampling_rate=sampling_rate)
    
    # Tachysystole detection
    tachy_result = detect_tachysystole(uc, sampling_rate=sampling_rate)
    
    # Sinusoidal pattern detection
    sinus_result = detect_sinusoidal_pattern(fhr, sampling_rate=sampling_rate)
    
    return {
        'baseline': baseline_result,
        'variability': variability_result,
        'decelerations': decelerations,
        'tachysystole': tachy_result,
        'sinusoidal': sinus_result
    }


def process_record(
    record_id: str,
    loader: CTUDataLoader,
    preprocessor: CTGPreprocessor,
    moment_extractor: MomentFeatureExtractor,
    config: DatasetConfig
) -> tuple[np.ndarray, int]:
    """
    Process a single CTG record through the full pipeline.
    
    Args:
        record_id: The record identifier.
        loader: CTU data loader instance.
        preprocessor: CTG preprocessor instance.
        moment_extractor: MOMENT feature extractor instance.
        config: Dataset configuration.
        
    Returns:
        Tuple of (feature_matrix, label) where feature_matrix has shape (N_windows, 1035).
    """
    # Load record
    record = loader.load_record(record_id)
    
    # Get label from pH
    label = loader.get_outcome_label(record_id)
    
    # Preprocess FHR
    prep_result = preprocessor.process(record.fhr1)
    fhr_clean = prep_result.processed_signal
    
    # Preprocess UC (or use raw if no preprocessing needed)
    uc_prep_result = preprocessor.process(record.uc)
    uc_clean = uc_prep_result.processed_signal
    
    # Run rule engine
    rules = run_rule_engine(fhr_clean, uc_clean, record.sampling_rate)
    
    # Extract MOMENT embeddings using sliding window
    embeddings = extract_embeddings_sliding_window(
        fhr=fhr_clean,
        extractor=moment_extractor,
        window_minutes=config.window_minutes,
        step_minutes=config.step_minutes,
        sampling_rate=config.sampling_rate
    )
    
    if len(embeddings) == 0:
        raise ValueError(f"No embeddings extracted for record {record_id}")
    
    # Build feature vectors (one per window), then matrix
    feature_vectors = []
    for emb_result in embeddings:
        fv = build_feature_vector(
            embedding=emb_result.embedding,
            baseline=rules['baseline'],
            variability=rules['variability'],
            decelerations=rules['decelerations'],
            tachysystole=rules['tachysystole'],
            sinusoidal=rules['sinusoidal'],
            start_idx=emb_result.start_idx,
            end_idx=emb_result.end_idx,
            start_time_sec=emb_result.start_time_sec,
            end_time_sec=emb_result.end_time_sec
        )
        feature_vectors.append(fv)
    
    feature_matrix = build_feature_matrix(feature_vectors)
    
    return feature_matrix, label


def prepare_dataset(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare the complete training dataset.
    
    Runs the full pipeline on all (or limited) records and saves X.npy and y.npy.
    
    Args:
        config: Dataset configuration.
        
    Returns:
        Tuple of (X, y) arrays.
    """
    logger.info("=" * 60)
    logger.info("SentinelFetal Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Record limit: {config.limit if config.limit else 'All'}")
    logger.info(f"Use mock MOMENT: {config.use_mock}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 60)
    
    # Initialize components
    loader = CTUDataLoader(config.data_dir)
    preprocessor = CTGPreprocessor(PreprocessingConfig())
    
    # CRITICAL: Initialize MOMENT with use_mock=False for real embeddings
    logger.info(f"Initializing MOMENT encoder (use_mock={config.use_mock})...")
    moment_extractor = MomentFeatureExtractor(
        device=config.device,
        use_mock=config.use_mock
    )
    logger.info(f"MOMENT device: {moment_extractor.device}")
    
    # Get record list
    records = loader.list_records()
    if config.limit:
        records = records[:config.limit]
    
    logger.info(f"Processing {len(records)} records...")
    
    # Process records
    all_features = []
    all_labels = []
    processed = 0
    failed = 0
    
    start_time = time.time()
    
    for i, record_id in enumerate(records):
        try:
            logger.info(f"[{i+1}/{len(records)}] Processing record {record_id}...")
            
            record_start = time.time()
            feature_matrix, label = process_record(
                record_id, loader, preprocessor, moment_extractor, config
            )
            record_time = time.time() - record_start
            
            # Expand labels to match number of windows
            n_windows = feature_matrix.shape[0]
            labels = np.full(n_windows, label, dtype=np.int64)
            
            all_features.append(feature_matrix)
            all_labels.append(labels)
            
            logger.info(
                f"  → {n_windows} windows, label={label}, "
                f"time={record_time:.1f}s"
            )
            processed += 1
            
        except Exception as e:
            logger.error(f"  → Failed: {e}")
            failed += 1
            continue
    
    total_time = time.time() - start_time
    
    if len(all_features) == 0:
        raise RuntimeError("No records processed successfully!")
    
    # Concatenate all features and labels
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    logger.info("=" * 60)
    logger.info("Processing Complete!")
    logger.info(f"Records processed: {processed}")
    logger.info(f"Records failed: {failed}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Label distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 2={np.sum(y==2)}")
    logger.info("=" * 60)
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    x_path = output_path / "X.npy"
    y_path = output_path / "y.npy"
    
    np.save(x_path, X)
    np.save(y_path, y)
    
    logger.info(f"Saved X to: {x_path}")
    logger.info(f"Saved y to: {y_path}")
    
    return X, y


def main():
    """Main entry point for the dataset preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare training dataset for SentinelFetal classifier"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0",
        help="Path to CTU-UHB dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for X.npy and y.npy"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for MOMENT model"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock MOMENT embeddings (FOR TESTING ONLY)"
    )
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        use_mock=args.use_mock,  # Default is False!
        device=args.device
    )
    
    try:
        X, y = prepare_dataset(config)
        logger.info("Dataset preparation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
