"""
Phase 4 Tests: Classifier & Safety Net.

This module tests:
1. pH extraction and label generation from loader
2. Medical override (safety net) logic
3. Dataset preparation pipeline integration
4. Classifier training basics
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.data.loader import CTUDataLoader
from src.rules.baseline import BaselineResult
from src.rules.variability import VariabilityResult, VariabilityCategory
from src.rules.decelerations import Deceleration, DecelerationType
from src.rules.tachysystole import TachysystoleResult
from src.rules.sinusoidal import SinusoidalResult
from src.analysis.override import apply_medical_override, OverrideReason


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def data_dir():
    """Path to CTU-UHB dataset."""
    return Path("data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0")


@pytest.fixture
def loader(data_dir):
    """CTU data loader instance."""
    return CTUDataLoader(str(data_dir))


@pytest.fixture
def normal_baseline():
    """Normal baseline result (130 bpm)."""
    return BaselineResult(
        value=130.0,
        is_normal=True,
        is_bradycardia=False,
        is_tachycardia=False,
        stable_segment_found=True,
        segment_start_idx=0,
        segment_variability=10.0,
        confidence=0.9
    )


@pytest.fixture
def bradycardia_baseline():
    """Bradycardia baseline result (100 bpm)."""
    return BaselineResult(
        value=100.0,
        is_normal=False,
        is_bradycardia=True,
        is_tachycardia=False,
        stable_segment_found=True,
        segment_start_idx=0,
        segment_variability=10.0,
        confidence=0.9
    )


@pytest.fixture
def moderate_variability():
    """Moderate variability result (normal)."""
    return VariabilityResult(
        value=12.0,
        category=VariabilityCategory.MODERATE,
        is_normal=True,
        is_concerning=False,
        window_values=[12.0, 11.0, 13.0],
        min_value=11.0,
        max_value=13.0,
        n_windows=3
    )


@pytest.fixture
def absent_variability():
    """Absent variability result (abnormal)."""
    return VariabilityResult(
        value=2.0,
        category=VariabilityCategory.ABSENT,
        is_normal=False,
        is_concerning=True,
        window_values=[2.0, 1.5, 2.5],
        min_value=1.5,
        max_value=2.5,
        n_windows=3
    )


@pytest.fixture
def sinusoidal_detected():
    """Sinusoidal pattern detected."""
    return SinusoidalResult(
        detected=True,
        confidence=0.95,
        dominant_frequency=0.05,
        frequency_cycles_per_min=3.0,
        amplitude=10.0,
        amplitude_in_range=True,
        dominance_ratio=0.6,
        analysis_duration_minutes=10.0
    )


@pytest.fixture
def sinusoidal_not_detected():
    """Sinusoidal pattern not detected."""
    return SinusoidalResult(
        detected=False,
        confidence=0.1,
        dominant_frequency=None,
        frequency_cycles_per_min=None,
        amplitude=0.0,
        amplitude_in_range=False,
        dominance_ratio=0.1,
        analysis_duration_minutes=10.0
    )


@pytest.fixture
def no_tachysystole():
    """No tachysystole."""
    return TachysystoleResult(
        detected=False,
        contractions_per_10min=4.0,
        total_contractions=4,
        analysis_duration_minutes=10.0,
        contraction_indices=[],
        confidence=0.9
    )


@pytest.fixture
def late_decelerations():
    """List with recurrent late decelerations."""
    return [
        Deceleration(
            start_idx=100, end_idx=200, nadir_idx=150,
            depth=30.0, duration_seconds=25.0,
            decel_type=DecelerationType.LATE, lag_seconds=25.0,
            has_severity_signs=False
        ),
        Deceleration(
            start_idx=500, end_idx=600, nadir_idx=550,
            depth=35.0, duration_seconds=25.0,
            decel_type=DecelerationType.LATE, lag_seconds=20.0,
            has_severity_signs=False
        ),
        Deceleration(
            start_idx=900, end_idx=1000, nadir_idx=950,
            depth=40.0, duration_seconds=25.0,
            decel_type=DecelerationType.LATE, lag_seconds=22.0,
            has_severity_signs=False
        ),
    ]


@pytest.fixture
def no_decelerations():
    """Empty deceleration list."""
    return []


# ==============================================================================
# Test pH Extraction
# ==============================================================================

class TestPHExtraction:
    """Tests for pH extraction from loader."""
    
    def test_extract_ph_record_1001(self, loader):
        """Test pH extraction for record 1001."""
        ph = loader.extract_ph("1001")
        assert ph is not None
        assert ph == pytest.approx(7.14, abs=0.01)
    
    def test_get_outcome_label_pathological(self, loader):
        """Test that pH < 7.15 gives label 2 (Pathological)."""
        label = loader.get_outcome_label("1001")  # pH = 7.14
        assert label == 2
    
    def test_get_outcome_label_unknown_returns_normal(self, loader):
        """Test that unknown pH returns label 0 (Normal)."""
        # Create a mock loader that returns None for pH
        class MockLoader(CTUDataLoader):
            def extract_ph(self, record_id):
                return None
        
        mock_loader = MockLoader(loader.data_dir)
        label = mock_loader.get_outcome_label("1001")
        assert label == 0
    
    def test_ph_label_boundaries(self, loader):
        """Test label boundary logic."""
        # Test the logic directly
        def get_label_from_ph(ph):
            if ph is None:
                return 0
            if ph < 7.15:
                return 2
            elif ph < 7.20:
                return 1
            else:
                return 0
        
        assert get_label_from_ph(7.10) == 2  # Pathological
        assert get_label_from_ph(7.14) == 2  # Pathological
        assert get_label_from_ph(7.15) == 1  # Intermediate (boundary)
        assert get_label_from_ph(7.17) == 1  # Intermediate
        assert get_label_from_ph(7.19) == 1  # Intermediate
        assert get_label_from_ph(7.20) == 0  # Normal (boundary)
        assert get_label_from_ph(7.25) == 0  # Normal
        assert get_label_from_ph(None) == 0  # Unknown → Normal


# ==============================================================================
# Test Medical Override (Safety Net)
# ==============================================================================

class TestMedicalOverride:
    """Tests for medical override safety net logic."""
    
    def test_sinusoidal_forces_category_3(
        self,
        normal_baseline,
        moderate_variability,
        sinusoidal_detected,
        no_tachysystole,
        no_decelerations
    ):
        """
        CRITICAL SAFETY TEST:
        Sinusoidal pattern MUST force Category 3 regardless of ML prediction.
        """
        # ML predicts Normal (0), but sinusoidal is detected
        ml_prediction = 0
        
        override = apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=normal_baseline,
            variability=moderate_variability,
            decelerations=no_decelerations,
            tachysystole=no_tachysystole,
            sinusoidal=sinusoidal_detected
        )
        
        assert override.should_override is True
        assert override.final_category == 2  # Category 3 (Pathological)
        assert override.reason == OverrideReason.SINUSOIDAL_PATTERN
        assert "Sinusoidal" in override.explanation
    
    def test_absent_variability_with_late_decels_forces_category_3(
        self,
        normal_baseline,
        absent_variability,
        sinusoidal_not_detected,
        no_tachysystole,
        late_decelerations
    ):
        """
        CRITICAL SAFETY TEST:
        Absent variability + recurrent late decels MUST force Category 3.
        """
        # ML predicts Intermediate (1), but absent variability + late decels
        ml_prediction = 1
        
        override = apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=normal_baseline,
            variability=absent_variability,
            decelerations=late_decelerations,
            tachysystole=no_tachysystole,
            sinusoidal=sinusoidal_not_detected
        )
        
        assert override.should_override is True
        assert override.final_category == 2  # Category 3 (Pathological)
        assert override.reason == OverrideReason.ABSENT_VARIABILITY_WITH_DECELS
        assert "Absent" in override.explanation or "variability" in override.explanation
    
    def test_absent_variability_with_bradycardia_forces_category_3(
        self,
        bradycardia_baseline,
        absent_variability,
        sinusoidal_not_detected,
        no_tachysystole,
        no_decelerations
    ):
        """
        CRITICAL SAFETY TEST:
        Absent variability + bradycardia MUST force Category 3.
        """
        ml_prediction = 0  # ML says Normal
        
        override = apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=bradycardia_baseline,
            variability=absent_variability,
            decelerations=no_decelerations,
            tachysystole=no_tachysystole,
            sinusoidal=sinusoidal_not_detected
        )
        
        assert override.should_override is True
        assert override.final_category == 2  # Category 3 (Pathological)
        assert override.reason == OverrideReason.ABSENT_VARIABILITY_WITH_DECELS
    
    def test_safety_floor_absent_variability_upgrades_normal_to_intermediate(
        self,
        normal_baseline,
        absent_variability,
        sinusoidal_not_detected,
        no_tachysystole,
        no_decelerations
    ):
        """
        Safety floor test:
        Absent variability with Normal prediction should upgrade to Intermediate.
        """
        ml_prediction = 0  # ML says Normal
        
        override = apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=normal_baseline,
            variability=absent_variability,
            decelerations=no_decelerations,
            tachysystole=no_tachysystole,
            sinusoidal=sinusoidal_not_detected
        )
        
        assert override.should_override is True
        assert override.final_category == 1  # Category 2 (Intermediate)
        assert override.reason == OverrideReason.ABSENT_VARIABILITY_SAFETY_FLOOR
    
    def test_no_override_when_all_normal(
        self,
        normal_baseline,
        moderate_variability,
        sinusoidal_not_detected,
        no_tachysystole,
        no_decelerations
    ):
        """Test that no override happens when everything is normal."""
        ml_prediction = 0  # ML says Normal
        
        override = apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=normal_baseline,
            variability=moderate_variability,
            decelerations=no_decelerations,
            tachysystole=no_tachysystole,
            sinusoidal=sinusoidal_not_detected
        )
        
        assert override.should_override is False
        assert override.final_category == 0  # Stays Normal
        assert override.reason == OverrideReason.NONE
    
    def test_ml_pathological_prediction_not_downgraded(
        self,
        normal_baseline,
        moderate_variability,
        sinusoidal_not_detected,
        no_tachysystole,
        no_decelerations
    ):
        """Test that ML pathological prediction is not downgraded."""
        ml_prediction = 2  # ML says Pathological
        
        override = apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=normal_baseline,
            variability=moderate_variability,
            decelerations=no_decelerations,
            tachysystole=no_tachysystole,
            sinusoidal=sinusoidal_not_detected
        )
        
        # Even though everything looks fine, ML said pathological
        # No override should DOWNGRADE the prediction
        assert override.final_category == 2  # Stays Pathological


# ==============================================================================
# Test Dataset Preparation (Dry Run)
# ==============================================================================

class TestDatasetPreparation:
    """Integration tests for dataset preparation."""
    
    def test_processed_directory_exists_after_prepare(self, data_dir):
        """
        Test that prepare_data.py creates X.npy with correct shape.
        
        This test uses --limit 3 to keep it fast.
        """
        # Import here to avoid import errors during collection
        import subprocess
        import os
        
        # Get project root
        project_root = Path(__file__).parent.parent
        
        # Run prepare_data with mock mode (for faster testing)
        cmd = [
            sys.executable,
            str(project_root / "src" / "training" / "prepare_data.py"),
            "--limit", "3",
            "--use-mock"  # Use mock for faster testing
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        # Check it ran successfully
        assert result.returncode == 0, f"prepare_data.py failed: {result.stderr}"
        
        # Check X.npy was created
        x_path = project_root / "data" / "processed" / "X.npy"
        y_path = project_root / "data" / "processed" / "y.npy"
        
        assert x_path.exists(), "X.npy was not created"
        assert y_path.exists(), "y.npy was not created"
        
        # Load and check shapes
        X = np.load(x_path)
        y = np.load(y_path)
        
        assert X.ndim == 2, "X should be 2D"
        assert X.shape[1] == 1035, f"X should have 1035 features, got {X.shape[1]}"
        assert len(y) == X.shape[0], "X and y should have same number of samples"
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Label distribution: {np.bincount(y)}")


# ==============================================================================
# Test Classifier (Basic)
# ==============================================================================

class TestClassifier:
    """Basic tests for the XGBoost classifier."""
    
    def test_classifier_import(self):
        """Test that classifier can be imported."""
        try:
            from src.models.classifier import XGBClassifierWrapper, ClassifierConfig
            assert XGBClassifierWrapper is not None
            assert ClassifierConfig is not None
        except ImportError as e:
            pytest.skip(f"XGBoost or sklearn not installed: {e}")
    
    def test_classifier_instantiation(self):
        """Test that classifier can be instantiated."""
        try:
            from src.models.classifier import XGBClassifierWrapper
            classifier = XGBClassifierWrapper()
            assert classifier.is_trained is False
        except ImportError as e:
            pytest.skip(f"XGBoost or sklearn not installed: {e}")
    
    def test_classifier_training_synthetic(self):
        """Test classifier training on synthetic data."""
        try:
            from src.models.classifier import XGBClassifierWrapper
            
            # Create synthetic data
            np.random.seed(42)
            X = np.random.randn(100, 1035)
            y = np.random.randint(0, 3, size=100)
            
            classifier = XGBClassifierWrapper()
            result = classifier.train(X, y)
            
            assert classifier.is_trained is True
            assert result.mean_cv_score >= 0.0
            assert result.confusion_matrix is not None
            
            # Test prediction
            predictions = classifier.predict(X[:10])
            assert len(predictions) == 10
            assert all(p in [0, 1, 2] for p in predictions)
            
        except ImportError as e:
            pytest.skip(f"XGBoost or sklearn not installed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
