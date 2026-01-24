"""
Tests for Phase 5 - UI Components and Alert Engine.

This module tests the alert generation with exact Hebrew strings
and UI component functionality.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pytest
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


class TestAlertGeneration:
    """Tests for Alert Engine Hebrew strings."""
    
    def test_category_3_headline_exact_string(self):
        """Test Category 3 headline matches exact Hebrew string from spec."""
        from src.analysis.alerts import generate_alert
        from src.rules.variability import VariabilityCategory
        
        alert = generate_alert(
            category=3,
            confidence=0.92,
            baseline=90,
            variability={'value': 3, 'category': 'Absent'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        assert alert.headline == 'התראה אדומה - קטגוריה 3 (פתולוגי)'
    
    def test_category_2_headline_exact_string(self):
        """Test Category 2 headline matches exact Hebrew string from spec."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=2,
            confidence=0.85,
            baseline=145,
            variability={'value': 8, 'category': 'Minimal'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        assert alert.headline == 'התראה כתומה - קטגוריה 2 (ביניים)'
    
    def test_category_1_headline_exact_string(self):
        """Test Category 1 headline matches exact Hebrew string from spec."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=1,
            confidence=0.95,
            baseline=140,
            variability={'value': 12, 'category': 'Moderate'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        assert alert.headline == 'סטטוס ירוק - קטגוריה 1 (תקין)'
    
    def test_category_3_recommendations_exact_strings(self):
        """Test Category 3 recommendations match exact Hebrew strings from spec."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=3,
            confidence=0.92,
            baseline=90,
            variability={'value': 3, 'category': 'Absent'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': True}
        )
        
        expected_recommendations = [
            'הערכה מיידית של הסיבות האפשריות',
            'שקילת החייאה תוך-רחמית',
            'היערכות ליילוד מיידי אם אין שיפור'
        ]
        
        assert alert.recommendations == expected_recommendations
    
    def test_category_2_recommendations_exact_strings(self):
        """Test Category 2 recommendations match exact Hebrew strings from spec."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=2,
            confidence=0.85,
            baseline=145,
            variability={'value': 8, 'category': 'Minimal'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        expected_recommendations = [
            'המשך ניטור צמוד',
            'שינוי תנוחת היולדת',
            'בדיקת לחץ דם והתייבשות'
        ]
        
        assert alert.recommendations == expected_recommendations
    
    def test_category_1_recommendations_exact_strings(self):
        """Test Category 1 recommendations match exact Hebrew string from spec."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=1,
            confidence=0.95,
            baseline=140,
            variability={'value': 12, 'category': 'Moderate'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        assert alert.recommendations == ['המשך ניטור שגרתי']
    
    def test_bradycardia_finding_hebrew(self):
        """Test bradycardia finding uses correct Hebrew text."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=2,
            confidence=0.8,
            baseline=100,
            variability={'value': 10, 'category': 'Moderate'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        bradycardia_findings = [f for f in alert.findings if 'ברדיקרדיה' in f]
        assert len(bradycardia_findings) == 1
        assert 'קצב בסיסי 100 bpm' in bradycardia_findings[0]
    
    def test_tachycardia_finding_hebrew(self):
        """Test tachycardia finding uses correct Hebrew text."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=2,
            confidence=0.8,
            baseline=175,
            variability={'value': 10, 'category': 'Moderate'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        tachycardia_findings = [f for f in alert.findings if 'טכיקרדיה' in f]
        assert len(tachycardia_findings) == 1
        assert 'קצב בסיסי 175 bpm' in tachycardia_findings[0]
    
    def test_absent_variability_finding_hebrew(self):
        """Test absent variability finding uses correct Hebrew text."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=3,
            confidence=0.9,
            baseline=140,
            variability={'value': 2.5, 'category': 'Absent'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        absent_findings = [f for f in alert.findings if 'היעדר שונות' in f]
        assert len(absent_findings) == 1
        assert 'ממצא חמור' in absent_findings[0]
    
    def test_sinusoidal_finding_hebrew(self):
        """Test sinusoidal finding uses correct Hebrew text."""
        from src.analysis.alerts import generate_alert
        
        alert = generate_alert(
            category=3,
            confidence=0.95,
            baseline=140,
            variability={'value': 10, 'category': 'Moderate'},
            decelerations=[],
            tachysystole={'detected': False},
            sinusoidal={'detected': True}
        )
        
        sinus_findings = [f for f in alert.findings if 'סינוסואידלית' in f]
        assert len(sinus_findings) == 1
        assert 'ממצא חמור' in sinus_findings[0]
    
    def test_late_deceleration_finding_hebrew(self):
        """Test late deceleration finding uses correct Hebrew text."""
        from src.analysis.alerts import generate_alert
        from src.rules.decelerations import Deceleration, DecelerationType
        
        late_decel = Deceleration(
            start_idx=0,
            end_idx=120,
            nadir_idx=60,
            nadir_value=90,
            depth=40,
            duration_seconds=30,
            decel_type=DecelerationType.LATE,
            lag_seconds=20,
            has_severity_signs=False
        )
        
        alert = generate_alert(
            category=2,
            confidence=0.85,
            baseline=140,
            variability={'value': 10, 'category': 'Moderate'},
            decelerations=[late_decel],
            tachysystole={'detected': False},
            sinusoidal={'detected': False}
        )
        
        late_findings = [f for f in alert.findings if 'האטות מאוחרות' in f]
        assert len(late_findings) == 1
        assert '1' in late_findings[0]


class TestAlertDataclass:
    """Tests for Alert dataclass validation."""
    
    def test_alert_creation_with_valid_category(self):
        """Test Alert creation with valid categories."""
        from src.analysis.alerts import Alert
        
        for cat in [1, 2, 3]:
            alert = Alert(
                category=cat,
                confidence=0.9,
                headline=f"Test {cat}",
                explanation="Test"
            )
            assert alert.category == cat
    
    def test_alert_invalid_category_raises(self):
        """Test Alert raises on invalid category."""
        from src.analysis.alerts import Alert
        
        with pytest.raises(ValueError, match="Category must be 1, 2, or 3"):
            Alert(
                category=4,
                confidence=0.9,
                headline="Invalid",
                explanation="Test"
            )
    
    def test_alert_invalid_confidence_raises(self):
        """Test Alert raises on invalid confidence."""
        from src.analysis.alerts import Alert
        
        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            Alert(
                category=1,
                confidence=1.5,
                headline="Invalid",
                explanation="Test"
            )
    
    def test_alert_timestamp_auto_generated(self):
        """Test Alert timestamp is auto-generated."""
        from src.analysis.alerts import Alert
        
        alert = Alert(
            category=1,
            confidence=0.9,
            headline="Test",
            explanation="Test"
        )
        
        # Should be valid ISO format
        datetime.fromisoformat(alert.timestamp)


class TestCategoryHelpers:
    """Tests for category helper functions."""
    
    def test_get_category_color(self):
        """Test category color mapping."""
        from src.analysis.alerts import get_category_color
        
        assert get_category_color(1) == 'green'
        assert get_category_color(2) == 'orange'
        assert get_category_color(3) == 'red'
    
    def test_get_category_emoji(self):
        """Test category emoji mapping."""
        from src.analysis.alerts import get_category_emoji
        
        assert get_category_emoji(1) == '🟢'
        assert get_category_emoji(2) == '🟠'
        assert get_category_emoji(3) == '🔴'


class TestTrainDemoScript:
    """Tests for train_demo.py script."""
    
    def test_train_demo_script_exists(self):
        """Test train_demo.py script exists."""
        train_demo_path = Path(__file__).parent.parent / "src" / "training" / "train_demo.py"
        assert train_demo_path.exists()
    
    def test_train_demo_imports(self):
        """Test train_demo.py can be imported."""
        from src.training.train_demo import train_demo_model
        assert callable(train_demo_model)


class TestAlertsModuleImports:
    """Tests for alerts module imports."""

    def test_alerts_module_imports(self):
        """Test alerts.py can be imported."""
        from src.analysis import alerts
        assert hasattr(alerts, 'Alert')
        assert hasattr(alerts, 'generate_alert')
        assert hasattr(alerts, 'get_category_color')
        assert hasattr(alerts, 'get_category_emoji')


class TestStateBridgeImports:
    """Tests for state_bridge module imports (V3 architecture)."""

    def test_state_bridge_imports_from_interfaces(self):
        """Test state_bridge can be imported from src.interfaces."""
        from src.interfaces.state_bridge import (
            get_data_bridge,
            DataBridge,
            PatientSnapshot,
            WardSnapshot,
            HighlightRegion,
        )
        assert callable(get_data_bridge)
        assert DataBridge is not None
        assert PatientSnapshot is not None

    def test_state_bridge_backwards_compat(self):
        """Test backwards compatibility import from src.ui."""
        from src.ui import (
            get_data_bridge,
            DataBridge,
            PatientSnapshot,
        )
        assert callable(get_data_bridge)
