"""
Visual Validation of CTG Preprocessing

Loads a record from CTU-UHB dataset and displays a comparison
of raw vs processed signals.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import CTUDataLoader
from data.preprocess import CTGPreprocessor, PreprocessingConfig


def create_synthetic_demo_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic CTG data for demonstration.
    
    Returns:
        Tuple of (timestamps, fhr_signal)
    """
    np.random.seed(42)
    
    # Create 10 minutes of data at 4 Hz
    duration_seconds = 600
    sampling_rate = 4
    n_samples = duration_seconds * sampling_rate
    
    timestamps = np.arange(n_samples) / sampling_rate
    
    # Base FHR around 140 BPM with natural variability
    base_fhr = 140 + 10 * np.sin(2 * np.pi * timestamps / 300)  # Slow oscillation
    variability = np.random.normal(0, 3, n_samples)  # Normal variability
    fhr = base_fhr + variability
    
    # Add some accelerations
    for start in [500, 1200, 1800]:
        duration = 80  # 20 seconds
        peak = 25
        for i in range(duration):
            if start + i < n_samples:
                # Gaussian-shaped acceleration
                t = (i - duration/2) / (duration/4)
                fhr[start + i] += peak * np.exp(-t**2)
    
    # Add some decelerations  
    for start in [800, 1500]:
        duration = 60
        depth = 20
        for i in range(duration):
            if start + i < n_samples:
                t = (i - duration/2) / (duration/4)
                fhr[start + i] -= depth * np.exp(-t**2)
    
    # Create gaps of various sizes
    # 5-second gap (should be filled)
    gap1_start = 400
    gap1_samples = 20  # 5 seconds
    fhr[gap1_start:gap1_start + gap1_samples] = np.nan
    
    # 15-second gap (should remain NaN)
    gap2_start = 1000
    gap2_samples = 60  # 15 seconds
    fhr[gap2_start:gap2_start + gap2_samples] = np.nan
    
    # Another 8-second gap (should be filled)
    gap3_start = 1600
    gap3_samples = 32  # 8 seconds
    fhr[gap3_start:gap3_start + gap3_samples] = np.nan
    
    # Add some out-of-range values
    fhr[200:205] = 45   # Below 50 BPM
    fhr[600:603] = 250  # Above 240 BPM
    fhr[900:902] = 0    # Zero (signal loss)
    
    return timestamps, fhr


def visualize_preprocessing(
    timestamps: np.ndarray,
    raw_fhr: np.ndarray,
    processed_fhr: np.ndarray,
    filled_mask: np.ndarray,
    unfilled_mask: np.ndarray,
    save_path: str = None
):
    """
    Create visualization comparing raw and processed FHR signals.
    
    Args:
        timestamps: Time values in seconds
        raw_fhr: Original FHR signal
        processed_fhr: Preprocessed FHR signal
        filled_mask: Boolean mask for filled gaps
        unfilled_mask: Boolean mask for unfilled gaps
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('CTG Preprocessing Validation', fontsize=14, fontweight='bold')
    
    # Convert timestamps to minutes for readability
    time_minutes = timestamps / 60
    
    # --- Plot 1: Raw Signal ---
    ax1 = axes[0]
    ax1.plot(time_minutes, raw_fhr, 'b-', linewidth=0.8, alpha=0.8, label='Raw FHR')
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Valid range (50-240)')
    ax1.axhline(y=240, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=110, color='g', linestyle=':', alpha=0.5, label='Normal range (110-160)')
    ax1.axhline(y=160, color='g', linestyle=':', alpha=0.5)
    ax1.set_ylabel('FHR (BPM)')
    ax1.set_title('Raw Signal (with gaps and out-of-range values)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 260)
    ax1.grid(True, alpha=0.3)
    
    # Highlight NaN regions in raw signal
    nan_regions = np.isnan(raw_fhr) | (raw_fhr < 50) | (raw_fhr > 240) | (raw_fhr == 0)
    ax1.fill_between(time_minutes, 0, 260, where=nan_regions, 
                     color='red', alpha=0.2, label='Invalid data')
    
    # --- Plot 2: Processed Signal ---
    ax2 = axes[1]
    ax2.plot(time_minutes, processed_fhr, 'g-', linewidth=0.8, alpha=0.8, label='Processed FHR')
    ax2.axhline(y=110, color='g', linestyle=':', alpha=0.5)
    ax2.axhline(y=160, color='g', linestyle=':', alpha=0.5)
    ax2.set_ylabel('FHR (BPM)')
    ax2.set_title('Processed Signal (10-second rule applied)')
    ax2.set_ylim(100, 180)
    ax2.grid(True, alpha=0.3)
    
    # Highlight filled regions
    ax2.fill_between(time_minutes, 100, 180, where=filled_mask,
                     color='blue', alpha=0.3, label='Filled gaps (≤10s)')
    
    # Highlight unfilled regions
    ax2.fill_between(time_minutes, 100, 180, where=unfilled_mask,
                     color='red', alpha=0.3, label='Unfilled gaps (>10s)')
    
    ax2.legend(loc='upper right')
    
    # --- Plot 3: Comparison ---
    ax3 = axes[2]
    ax3.plot(time_minutes, raw_fhr, 'b-', linewidth=0.8, alpha=0.5, label='Raw')
    ax3.plot(time_minutes, processed_fhr, 'g-', linewidth=1.2, alpha=0.8, label='Processed')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('FHR (BPM)')
    ax3.set_title('Raw vs Processed Comparison')
    ax3.set_ylim(100, 180)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def print_preprocessing_stats(result):
    """Print preprocessing statistics."""
    stats = result.stats
    print("\n" + "="*50)
    print("PREPROCESSING STATISTICS")
    print("="*50)
    print(f"Total samples:        {stats['total_samples']:,}")
    print(f"Duration:             {stats['duration_seconds']:.1f} seconds ({stats['duration_seconds']/60:.1f} minutes)")
    print(f"Invalid samples:      {stats['invalid_samples']:,} ({stats['invalid_percent']:.1f}%)")
    print(f"Filled samples:       {stats['filled_samples']:,} ({stats['filled_percent']:.1f}%)")
    print(f"Remaining NaN:        {stats['remaining_nan']:,} ({stats['remaining_nan_percent']:.1f}%)")
    print(f"Mean FHR:             {stats['mean_fhr']:.1f} BPM")
    print(f"Std FHR:              {stats['std_fhr']:.1f} BPM")
    print("="*50)


def main():
    """Main function to run the visualization."""
    print("="*60)
    print("SentinelFetal - Preprocessing Visualization")
    print("="*60)
    
    # Try to load real data first
    data_dir = Path(__file__).parent.parent / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
    
    use_synthetic = True
    
    if data_dir.exists():
        try:
            print(f"\nLooking for CTU-UHB data in: {data_dir}")
            loader = CTUDataLoader(data_dir)
            records = loader.list_records()
            
            if records:
                print(f"Found {len(records)} records")
                record = loader.load_record(records[0])
                timestamps = record.timestamps
                raw_fhr = record.fhr1.astype(float)
                use_synthetic = False
                print(f"Loaded record: {record.record_id}")
        except Exception as e:
            print(f"Could not load real data: {e}")
    
    if use_synthetic:
        print("\nUsing synthetic demo data for visualization...")
        print("(To use real data, download CTU-UHB dataset to 'data/ctu-uhb/')")
        timestamps, raw_fhr = create_synthetic_demo_data()
    
    # Preprocess
    print("\nPreprocessing signal...")
    config = PreprocessingConfig(
        sampling_rate=4.0,
        max_gap_seconds=10.0,
        fhr_min=50.0,
        fhr_max=240.0
    )
    preprocessor = CTGPreprocessor(config)
    result = preprocessor.process(raw_fhr)
    
    # Print statistics
    print_preprocessing_stats(result)
    
    # Visualize
    print("\nGenerating visualization...")
    output_path = Path(__file__).parent.parent / "preprocessing_validation.png"
    
    visualize_preprocessing(
        timestamps=timestamps,
        raw_fhr=result.original_signal,
        processed_fhr=result.processed_signal,
        filled_mask=result.filled_mask,
        unfilled_mask=result.unfilled_mask,
        save_path=str(output_path)
    )
    
    print("\n✅ Visualization complete!")
    print("\nKey observations:")
    print("  • Blue shaded areas in Plot 1: Invalid/missing data in raw signal")
    print("  • Blue shaded areas in Plot 2: Gaps ≤10s that were FILLED")
    print("  • Red shaded areas in Plot 2: Gaps >10s that remain as NaN")
    print("  • Plot 3: Direct comparison showing interpolation quality")


if __name__ == "__main__":
    main()
