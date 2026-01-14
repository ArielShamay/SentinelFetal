"""
Pipeline Integration Tests.

End-to-end tests validating the complete SentinelFetal Gen3.5 pipeline:
    1. Data Loading (CTU-UHB database)
    2. Preprocessing (gap filling, spike detection)
    3. Rule Engine (baseline, variability, decelerations, etc.)
    4. MOMENT Encoder (embedding extraction)
    5. Feature Fusion (1035-dimensional vector)

These tests verify that all components work together correctly.

References:
    SentinelFetal Gen3.5 Technical Specification, Sections 3-6
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import CTUDataLoader, DataLoaderError
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules import (
    calculate_baseline,
    calculate_variability,
    detect_decelerations,
    detect_tachysystole,
    detect_sinusoidal_pattern,
)
from src.models.moment_encoder import (
    MomentFeatureExtractor,
    extract_embeddings_sliding_window,
    EmbeddingResult,
    MOMENT_AVAILABLE,
)
from src.models.fusion import (
    build_feature_vector,
    build_feature_matrix,
    FeatureVector,
    FEATURE_VECTOR_DIM,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def data_dir() -> Path:
    """Path to the CTU-UHB dataset."""
    # Try multiple possible locations
    possible_paths = [
        Path("data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0"),
        Path("../data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0"),
        Path(__file__).parent.parent / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    pytest.skip("CTU-UHB dataset not found. Please download from PhysioNet.")


@pytest.fixture
def loader(data_dir: Path) -> CTUDataLoader:
    """Create data loader instance."""
    return CTUDataLoader(data_dir)


@pytest.fixture
def preprocessor() -> CTGPreprocessor:
    """Create preprocessor with default config."""
    config = PreprocessingConfig(
        sampling_rate=4.0,
        max_gap_seconds=10.0,
        fhr_min=50.0,
        fhr_max=240.0
    )
    return CTGPreprocessor(config)


@pytest.fixture
def moment_extractor() -> MomentFeatureExtractor:
    """Create MOMENT feature extractor (may use mock mode)."""
    return MomentFeatureExtractor(use_mock=True)  # Use mock for testing


# ============================================================================
# Test Classes
# ============================================================================

class TestDataLoading:
    """Test data loading from CTU-UHB database."""
    
    def test_load_record_1001(self, loader: CTUDataLoader):
        """Test loading record 1001."""
        record = loader.load_record("1001")
        
        assert record is not None
        assert record.record_id == "1001"
        assert len(record.fhr1) > 0
        assert record.sampling_rate == 4.0
        assert record.duration_seconds > 0
    
    def test_record_has_fhr_and_uc(self, loader: CTUDataLoader):
        """Test that record contains FHR and UC signals."""
        record = loader.load_record("1001")
        
        assert record.fhr1 is not None
        assert record.uc is not None
        assert len(record.fhr1) == len(record.uc)
    
    def test_list_records(self, loader: CTUDataLoader):
        """Test listing available records."""
        records = loader.list_records()
        
        assert len(records) > 0
        assert "1001" in records


class TestPreprocessing:
    """Test preprocessing pipeline."""
    
    def test_preprocess_record_1001(
        self, 
        loader: CTUDataLoader, 
        preprocessor: CTGPreprocessor
    ):
        """Test preprocessing on record 1001."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        assert result is not None
        assert len(result.processed_signal) == len(record.fhr1)
        assert result.stats['total_samples'] > 0
    
    def test_preprocessing_handles_gaps(
        self, 
        loader: CTUDataLoader, 
        preprocessor: CTGPreprocessor
    ):
        """Test that preprocessing fills gaps correctly."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        # Some gaps should have been filled
        # (most real records have some missing data)
        assert 'filled_samples' in result.stats


class TestRuleEngine:
    """Test rule engine components on real data."""
    
    def test_baseline_calculation(
        self, 
        loader: CTUDataLoader, 
        preprocessor: CTGPreprocessor
    ):
        """Test baseline calculation on preprocessed signal."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        baseline = calculate_baseline(
            result.processed_signal, 
            sampling_rate=4.0
        )
        
        assert baseline is not None
        assert 50 <= baseline.value <= 240  # Valid FHR range
        assert baseline.value % 5 == 0  # Rounded to nearest 5
    
    def test_variability_calculation(
        self, 
        loader: CTUDataLoader, 
        preprocessor: CTGPreprocessor
    ):
        """Test variability calculation."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        baseline = calculate_baseline(result.processed_signal)
        
        variability = calculate_variability(
            result.processed_signal,
            sampling_rate=4.0
        )
        
        assert variability is not None
        assert variability.value >= 0
        assert variability.category is not None
    
    def test_deceleration_detection(
        self, 
        loader: CTUDataLoader, 
        preprocessor: CTGPreprocessor
    ):
        """Test deceleration detection."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        baseline = calculate_baseline(result.processed_signal)
        
        decelerations = detect_decelerations(
            result.processed_signal,
            record.uc,
            baseline.value,
            sampling_rate=4.0
        )
        
        assert isinstance(decelerations, list)
        # Decelerations may or may not be present
    
    def test_tachysystole_detection(
        self, 
        loader: CTUDataLoader, 
        preprocessor: CTGPreprocessor
    ):
        """Test tachysystole detection."""
        record = loader.load_record("1001")
        
        tachysystole = detect_tachysystole(
            record.uc,
            sampling_rate=4.0
        )
        
        assert tachysystole is not None
        assert isinstance(tachysystole.detected, bool)
    
    def test_sinusoidal_detection(
        self, 
        loader: CTUDataLoader, 
        preprocessor: CTGPreprocessor
    ):
        """Test sinusoidal pattern detection."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        sinusoidal = detect_sinusoidal_pattern(
            result.processed_signal,
            sampling_rate=4.0
        )
        
        assert sinusoidal is not None
        assert isinstance(sinusoidal.detected, bool)


class TestMomentEncoder:
    """Test MOMENT encoder."""
    
    def test_moment_extractor_initialization(self, moment_extractor):
        """Test that extractor initializes correctly."""
        assert moment_extractor is not None
        # In mock mode, model is not loaded
        assert moment_extractor.use_mock or moment_extractor._model is not None
    
    def test_single_window_extraction(
        self, 
        loader: CTUDataLoader,
        preprocessor: CTGPreprocessor,
        moment_extractor: MomentFeatureExtractor
    ):
        """Test extracting embedding from a single window."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        # Take first 10 minutes (2400 samples @ 4Hz)
        window = result.processed_signal[:2400]
        
        embedding = moment_extractor.extract(window)
        
        assert embedding is not None
        assert embedding.shape == (1024,)
        assert not np.any(np.isnan(embedding))
    
    def test_sliding_window_extraction(
        self, 
        loader: CTUDataLoader,
        preprocessor: CTGPreprocessor,
        moment_extractor: MomentFeatureExtractor
    ):
        """Test sliding window embedding extraction."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        embeddings = extract_embeddings_sliding_window(
            result.processed_signal,
            moment_extractor,
            sampling_rate=4.0,
            window_minutes=10.0,
            step_minutes=1.0
        )
        
        assert len(embeddings) > 0
        
        for emb_result in embeddings:
            assert isinstance(emb_result, EmbeddingResult)
            assert emb_result.embedding.shape == (1024,)
    
    def test_short_signal_handling(self, moment_extractor: MomentFeatureExtractor):
        """Test handling of signals shorter than window size."""
        short_signal = np.random.randn(1000) * 10 + 140  # 1000 samples
        
        embedding = moment_extractor.extract(short_signal)
        
        assert embedding.shape == (1024,)


class TestFeatureFusion:
    """Test feature fusion module."""
    
    def test_build_feature_vector_dimensions(
        self, 
        loader: CTUDataLoader,
        preprocessor: CTGPreprocessor,
        moment_extractor: MomentFeatureExtractor
    ):
        """Test that feature vector has correct dimensions (1035)."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        # Extract components
        window = result.processed_signal[:2400]
        embedding = moment_extractor.extract(window)
        baseline = calculate_baseline(window)
        variability = calculate_variability(window, baseline.value)
        decelerations = detect_decelerations(window, record.uc[:2400], baseline.value)
        tachysystole = detect_tachysystole(record.uc[:2400])
        sinusoidal = detect_sinusoidal_pattern(window)
        
        # Build feature vector
        feature_vec = build_feature_vector(
            embedding=embedding,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )
        
        assert feature_vec is not None
        assert feature_vec.vector.shape == (FEATURE_VECTOR_DIM,)
        assert feature_vec.vector.shape == (1035,)
    
    def test_feature_vector_content(
        self, 
        loader: CTUDataLoader,
        preprocessor: CTGPreprocessor,
        moment_extractor: MomentFeatureExtractor
    ):
        """Test that feature vector contains expected values."""
        record = loader.load_record("1001")
        result = preprocessor.process(record.fhr1)
        
        window = result.processed_signal[:2400]
        embedding = moment_extractor.extract(window)
        baseline = calculate_baseline(window)
        variability = calculate_variability(window, baseline.value)
        decelerations = detect_decelerations(window, record.uc[:2400], baseline.value)
        tachysystole = detect_tachysystole(record.uc[:2400])
        sinusoidal = detect_sinusoidal_pattern(window)
        
        feature_vec = build_feature_vector(
            embedding=embedding,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )
        
        # Check embedding section
        assert np.allclose(feature_vec.vector[:1024], embedding)
        
        # Check baseline (normalized)
        assert feature_vec.vector[1024] == pytest.approx(baseline.value / 160.0, rel=0.01)
        
        # Check variability one-hot sums to 1
        var_onehot = feature_vec.vector[1026:1030]
        assert np.sum(var_onehot) == pytest.approx(1.0)


class TestFullPipeline:
    """End-to-end pipeline integration tests."""
    
    def test_complete_pipeline_record_1001(
        self, 
        loader: CTUDataLoader,
        preprocessor: CTGPreprocessor,
        moment_extractor: MomentFeatureExtractor
    ):
        """
        THE MAIN INTEGRATION TEST
        
        Complete pipeline test on record 1001:
        1. Load real CTU-UHB data
        2. Preprocess signal
        3. Extract MOMENT embeddings (sliding window)
        4. Run all rule engine components
        5. Build fused feature vectors
        6. Verify final output shape
        """
        # =====================================================================
        # Step 1: Load Data
        # =====================================================================
        record = loader.load_record("1001")
        assert record is not None
        print(f"\n[1/5] Loaded record 1001: {record.duration_seconds/60:.1f} minutes")
        
        # =====================================================================
        # Step 2: Preprocess
        # =====================================================================
        preprocess_result = preprocessor.process(record.fhr1)
        fhr = preprocess_result.processed_signal
        uc = record.uc
        print(f"[2/5] Preprocessed: {preprocess_result.stats['filled_percent']:.1f}% gaps filled")
        
        # =====================================================================
        # Step 3: Extract MOMENT Embeddings (Sliding Window)
        # =====================================================================
        embedding_results = extract_embeddings_sliding_window(
            fhr,
            moment_extractor,
            sampling_rate=4.0,
            window_minutes=10.0,
            step_minutes=1.0
        )
        n_windows = len(embedding_results)
        print(f"[3/5] Extracted {n_windows} MOMENT embeddings")
        
        assert n_windows > 0, "Should extract at least one embedding"
        
        # =====================================================================
        # Step 4 & 5: Rule Engine + Fusion for each window
        # =====================================================================
        feature_vectors = []
        
        for emb_result in embedding_results:
            # Get the window data
            start = emb_result.start_idx
            end = emb_result.end_idx
            fhr_window = fhr[start:end]
            uc_window = uc[start:end] if end <= len(uc) else uc[start:]
            
            # Rule Engine
            baseline = calculate_baseline(fhr_window)
            variability = calculate_variability(fhr_window)
            decelerations = detect_decelerations(fhr_window, uc_window, baseline.value)
            tachysystole = detect_tachysystole(uc_window)
            sinusoidal = detect_sinusoidal_pattern(fhr_window)
            
            # Fusion
            feature_vec = build_feature_vector(
                embedding=emb_result.embedding,
                baseline=baseline,
                variability=variability,
                decelerations=decelerations,
                tachysystole=tachysystole,
                sinusoidal=sinusoidal,
                start_idx=start,
                end_idx=end,
                start_time_sec=emb_result.start_time_sec,
                end_time_sec=emb_result.end_time_sec,
            )
            
            feature_vectors.append(feature_vec)
        
        print(f"[4/5] Built {len(feature_vectors)} feature vectors")
        
        # =====================================================================
        # Step 6: Build Feature Matrix and Verify Shape
        # =====================================================================
        feature_matrix = build_feature_matrix(feature_vectors)
        
        print(f"[5/5] Final feature matrix shape: {feature_matrix.shape}")
        
        # THE CRITICAL ASSERTION
        assert feature_matrix.shape == (n_windows, 1035), (
            f"Expected shape ({n_windows}, 1035), got {feature_matrix.shape}"
        )
        
        # Additional validations
        assert not np.any(np.isnan(feature_matrix)), "Feature matrix should not contain NaN"
        assert feature_matrix.dtype == np.float32, "Feature matrix should be float32"
        
        print("\n✅ PIPELINE INTEGRATION TEST PASSED!")
        print(f"   - Processed {record.duration_seconds/60:.1f} minutes of CTG data")
        print(f"   - Generated {n_windows} windows with 1035-dim feature vectors")
        print(f"   - Total features: {n_windows * 1035:,}")
    
    def test_pipeline_multiple_records(
        self, 
        loader: CTUDataLoader,
        preprocessor: CTGPreprocessor,
        moment_extractor: MomentFeatureExtractor
    ):
        """Test pipeline on multiple records to ensure consistency."""
        test_records = ["1001", "1002", "1003"]
        
        for record_id in test_records:
            try:
                record = loader.load_record(record_id)
                result = preprocessor.process(record.fhr1)
                
                # Quick extraction (first window only)
                window = result.processed_signal[:2400]
                if len(window) < 2400:
                    continue
                    
                embedding = moment_extractor.extract(window)
                baseline = calculate_baseline(window)
                variability = calculate_variability(window)
                decelerations = detect_decelerations(
                    window, record.uc[:2400], baseline.value
                )
                tachysystole = detect_tachysystole(record.uc[:2400])
                sinusoidal = detect_sinusoidal_pattern(window)
                
                feature_vec = build_feature_vector(
                    embedding=embedding,
                    baseline=baseline,
                    variability=variability,
                    decelerations=decelerations,
                    tachysystole=tachysystole,
                    sinusoidal=sinusoidal,
                )
                
                assert feature_vec.vector.shape == (1035,), (
                    f"Record {record_id}: Expected shape (1035,), "
                    f"got {feature_vec.vector.shape}"
                )
                
            except Exception as e:
                pytest.fail(f"Record {record_id} failed: {e}")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_deceleration_list(self, moment_extractor: MomentFeatureExtractor):
        """Test fusion with no decelerations."""
        embedding = np.random.randn(1024).astype(np.float32)
        
        from src.rules import BaselineResult, VariabilityResult, VariabilityCategory
        from src.rules import TachysystoleResult, SinusoidalResult
        
        baseline = BaselineResult(
            value=140.0,
            is_normal=True,
            is_bradycardia=False,
            is_tachycardia=False,
            stable_segment_found=True
        )
        
        variability = VariabilityResult(
            value=12.0,
            category=VariabilityCategory.MODERATE,
            is_normal=True,
            is_concerning=False,
            window_values=[10.0, 12.0, 14.0],
            min_value=8.0,
            max_value=18.0,
            n_windows=3
        )
        
        tachysystole = TachysystoleResult(
            detected=False,
            contractions_per_10min=3.0,
            total_contractions=9,
            analysis_duration_minutes=30.0,
            contraction_indices=[100, 500, 900, 1300, 1700, 2100, 2500, 2900, 3300]
        )
        
        sinusoidal = SinusoidalResult(
            detected=False,
            confidence=0.1,
            dominant_frequency=None,
            frequency_cycles_per_min=None,
            amplitude=5.0,
            amplitude_in_range=True,
            dominance_ratio=0.1,
            analysis_duration_minutes=20.0
        )
        
        feature_vec = build_feature_vector(
            embedding=embedding,
            baseline=baseline,
            variability=variability,
            decelerations=[],  # Empty list
            tachysystole=tachysystole,
            sinusoidal=sinusoidal,
        )
        
        assert feature_vec.vector.shape == (1035,)
        assert feature_vec.late_decel_count == 0
        assert feature_vec.variable_decel_count == 0
    
    def test_all_flags_active(self, moment_extractor: MomentFeatureExtractor):
        """Test fusion with all pathological flags active."""
        embedding = np.random.randn(1024).astype(np.float32)
        
        from src.rules import BaselineResult, VariabilityResult, VariabilityCategory
        from src.rules import TachysystoleResult, SinusoidalResult
        from src.rules import Deceleration, DecelerationType
        
        baseline = BaselineResult(
            value=100.0,  # Bradycardia
            is_normal=False,
            is_bradycardia=True,
            is_tachycardia=False,
            stable_segment_found=True
        )
        
        variability = VariabilityResult(
            value=1.0,
            category=VariabilityCategory.ABSENT,  # Pathological
            is_normal=False,
            is_concerning=True,
            window_values=[0.5, 1.0, 1.5],
            min_value=0.5,
            max_value=2.0,
            n_windows=3
        )
        
        # Multiple late decelerations
        decelerations = [
            Deceleration(
                start_idx=100,
                end_idx=200,
                nadir_idx=150,
                depth=30.0,
                duration_seconds=25.0,
                decel_type=DecelerationType.LATE,
                lag_seconds=20.0,
                has_severity_signs=False,
                severity_signs=[]
            ),
            Deceleration(
                start_idx=300,
                end_idx=400,
                nadir_idx=350,
                depth=25.0,
                duration_seconds=25.0,
                decel_type=DecelerationType.LATE,
                lag_seconds=18.0,
                has_severity_signs=False,
                severity_signs=[]
            ),
        ]
        
        tachysystole = TachysystoleResult(
            detected=True,  # Pathological
            contractions_per_10min=7.0,
            total_contractions=21,
            analysis_duration_minutes=30.0,
            contraction_indices=list(range(0, 21000, 1000))  # 21 contractions
        )
        
        sinusoidal = SinusoidalResult(
            detected=True,  # SEVERE - Category 3
            confidence=0.9,
            dominant_frequency=0.067,
            frequency_cycles_per_min=4.0,
            amplitude=10.0,
            amplitude_in_range=True,
            dominance_ratio=0.5,
            analysis_duration_minutes=25.0
        )
        
        feature_vec = build_feature_vector(
            embedding=embedding,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal,
            total_contractions=4,  # Recurrent (2/4 = 50%)
        )
        
        assert feature_vec.vector.shape == (1035,)
        assert feature_vec.baseline == 100.0
        assert feature_vec.variability_category == "ABSENT"
        assert feature_vec.late_decel_count == 2
        assert feature_vec.tachysystole is True
        assert feature_vec.sinusoidal is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
