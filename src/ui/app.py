"""
SentinelFetal Dashboard - Streamlit Application.

This is the main user interface for the SentinelFetal monitoring system.
It provides real-time CTG visualization and alert generation.

Usage:
    streamlit run src/ui/app.py

References:
    SentinelFetal Gen3.5 Technical Specification, Section 9
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import numpy as np
import streamlit as st

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.data.loader import CTUDataLoader, CTGRecord
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules.baseline import calculate_baseline
from src.rules.variability import calculate_variability, VariabilityCategory
from src.rules.decelerations import detect_decelerations, DecelerationType
from src.rules.tachysystole import detect_tachysystole
from src.rules.sinusoidal import detect_sinusoidal_pattern
from src.models.moment_encoder import MomentFeatureExtractor, extract_embeddings_sliding_window
from src.models.fusion import build_feature_vector, build_feature_matrix
from src.models.classifier import XGBClassifierWrapper
from src.analysis.override import apply_medical_override
from src.analysis.alerts import generate_alert, get_category_emoji, get_category_color, Alert
from src.ui.plots import create_ctg_plot, create_category_indicator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = "data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0"
MODEL_PATH = "models/xgb_demo.json"
SAMPLING_RATE = 4.0


# ============================================================================
# Caching and State Management
# ============================================================================

@st.cache_resource
def load_data_loader():
    """Load and cache the data loader."""
    try:
        loader = CTUDataLoader(DATA_DIR)
        return loader
    except Exception as e:
        st.error(f"שגיאה בטעינת נתונים: {e}")
        return None


@st.cache_resource
def load_moment_encoder():
    """Load and cache the MOMENT encoder (mock mode for demo)."""
    return MomentFeatureExtractor(use_mock=True)  # Use mock for demo


@st.cache_resource
def load_classifier():
    """Load and cache the classifier if available."""
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        try:
            classifier = XGBClassifierWrapper()
            classifier.load_model(str(model_path))
            return classifier
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}")
            return None
    return None


def get_preprocessor():
    """Get preprocessor instance."""
    return CTGPreprocessor(PreprocessingConfig())


# ============================================================================
# Pipeline Functions
# ============================================================================

def run_full_pipeline(record: CTGRecord, use_ml: bool = True) -> Dict[str, Any]:
    """
    Run the full SentinelFetal pipeline on a record.
    
    Args:
        record: CTG record to process.
        use_ml: Whether to use ML classifier (requires trained model).
        
    Returns:
        Dictionary with all pipeline results.
    """
    results = {}
    
    # Step 1: Preprocessing
    preprocessor = get_preprocessor()
    fhr_result = preprocessor.process(record.fhr)
    uc_result = preprocessor.process(record.uc)
    
    fhr_clean = fhr_result.processed_signal
    uc_clean = uc_result.processed_signal
    
    results['fhr'] = fhr_clean
    results['uc'] = uc_clean
    results['preprocessing'] = {
        'fhr_stats': fhr_result.stats,
        'uc_stats': uc_result.stats
    }
    
    # Step 2: Rule Engine
    baseline_result = calculate_baseline(fhr_clean)
    variability_result = calculate_variability(fhr_clean)
    decelerations = detect_decelerations(fhr_clean, uc_clean, baseline_result.value)
    tachysystole_result = detect_tachysystole(uc_clean)
    sinusoidal_result = detect_sinusoidal_pattern(fhr_clean)
    
    results['baseline'] = baseline_result
    results['variability'] = variability_result
    results['decelerations'] = decelerations
    results['tachysystole'] = tachysystole_result
    results['sinusoidal'] = sinusoidal_result
    
    # Step 3: Classification
    classifier = load_classifier()
    
    if use_ml and classifier is not None:
        # Use ML + MOMENT
        moment = load_moment_encoder()
        embeddings = extract_embeddings_sliding_window(
            fhr=fhr_clean,
            extractor=moment,
            window_minutes=10,
            step_minutes=1,
            sampling_rate=SAMPLING_RATE
        )
        
        if embeddings:
            # Build feature vectors
            feature_vectors = []
            for emb_result in embeddings:
                fv = build_feature_vector(
                    embedding=emb_result.embedding,
                    baseline=baseline_result,
                    variability=variability_result,
                    decelerations=decelerations,
                    tachysystole=tachysystole_result,
                    sinusoidal=sinusoidal_result,
                    start_idx=emb_result.start_idx,
                    end_idx=emb_result.end_idx,
                    start_time_sec=emb_result.start_time_sec,
                    end_time_sec=emb_result.end_time_sec
                )
                feature_vectors.append(fv)
            
            feature_matrix = build_feature_matrix(feature_vectors)
            
            # Predict on all windows, take majority vote
            predictions = classifier.predict(feature_matrix)
            probabilities = classifier.predict_proba(feature_matrix)
            
            # Majority vote
            from collections import Counter
            vote_counts = Counter(predictions)
            ml_prediction = vote_counts.most_common(1)[0][0]
            
            # Average confidence
            confidence = float(np.mean(np.max(probabilities, axis=1)))
        else:
            ml_prediction = 0  # Default to normal
            confidence = 0.5
    else:
        # Rule-based fallback
        ml_prediction = _rule_based_classify(
            variability_result, decelerations, baseline_result, sinusoidal_result
        )
        confidence = 0.7
    
    results['ml_prediction'] = ml_prediction
    results['confidence'] = confidence
    
    # Step 4: Medical Override
    # Count contractions for override
    from scipy.signal import find_peaks
    uc_valid = uc_clean[~np.isnan(uc_clean)]
    if len(uc_valid) > 0:
        peaks, _ = find_peaks(uc_valid, distance=240, height=20)
        total_contractions = len(peaks)
    else:
        total_contractions = 0
    
    final_category = apply_medical_override(
        ml_prediction=ml_prediction,
        variability=variability_result,
        decelerations=decelerations,
        baseline=baseline_result,
        sinusoidal=sinusoidal_result,
        total_contractions=total_contractions
    )
    
    results['final_category'] = final_category + 1  # Convert to 1/2/3
    
    # Step 5: Generate Alert
    alert = generate_alert(
        category=final_category + 1,
        confidence=confidence,
        baseline=baseline_result,
        variability=variability_result,
        decelerations=decelerations,
        tachysystole=tachysystole_result,
        sinusoidal=sinusoidal_result
    )
    
    results['alert'] = alert
    
    return results


def _rule_based_classify(variability, decelerations, baseline, sinusoidal) -> int:
    """Simple rule-based classification fallback."""
    # Category 3 (Pathological) = 2
    if sinusoidal.detected:
        return 2
    
    if variability.category == VariabilityCategory.ABSENT:
        late_count = sum(1 for d in decelerations if d.decel_type == DecelerationType.LATE)
        if late_count > 0 or baseline.is_bradycardia:
            return 2
    
    # Category 2 (Intermediate) = 1
    if variability.category in (VariabilityCategory.MINIMAL, VariabilityCategory.MARKED):
        return 1
    
    late_count = sum(1 for d in decelerations if d.decel_type == DecelerationType.LATE)
    if late_count > 0:
        return 1
    
    # Category 1 (Normal) = 0
    return 0


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar(loader: CTUDataLoader) -> Optional[str]:
    """Render the sidebar with patient list."""
    st.sidebar.title("🏥 רשימת יולדות")
    st.sidebar.markdown("---")
    
    # Get available records
    try:
        records = loader.list_records()
    except Exception as e:
        st.sidebar.error(f"שגיאה: {e}")
        return None
    
    # Limit to first 20 records for demo
    demo_records = records[:20]
    
    st.sidebar.info(f"מציג {len(demo_records)} מתוך {len(records)} רשומות")
    
    # Record selection
    selected = st.sidebar.selectbox(
        "בחרי רשומה:",
        options=demo_records,
        format_func=lambda x: f"מיטה {x}"
    )
    
    st.sidebar.markdown("---")
    
    # Options
    st.sidebar.subheader("⚙️ הגדרות")
    use_ml = st.sidebar.checkbox("השתמש במודל ML", value=True)
    st.session_state['use_ml'] = use_ml
    
    # Info
    st.sidebar.markdown("---")
    st.sidebar.caption("SentinelFetal Gen3.5")
    st.sidebar.caption("© 2026 אוניברסיטת אריאל")
    
    return selected


def render_patient_detail(results: Dict[str, Any], record_id: str):
    """Render patient details with CTG graph and findings."""
    alert: Alert = results['alert']
    
    # Header with category color
    category = results['final_category']
    emoji = get_category_emoji(category)
    color = get_category_color(category)
    
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">{emoji} {alert.headline}</h2>
            <p style="color: white; margin: 5px 0 0 0;">רשומה: {record_id} | ביטחון: {results['confidence']:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Two columns: Graph and Findings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 גרף CTG")
        fig = create_ctg_plot(
            fhr=results['fhr'],
            uc=results['uc'],
            decelerations=results['decelerations'],
            sampling_rate=SAMPLING_RATE,
            title=f"ניטור מיטה {record_id}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Findings panel
        st.subheader("🔍 ממצאים")
        for finding in alert.findings:
            st.write(f"• {finding}")
        
        st.markdown("---")
        
        # Recommendations panel
        st.subheader("💡 המלצות")
        for rec in alert.recommendations:
            if category == 3:
                st.error(f"• {rec}")
            elif category == 2:
                st.warning(f"• {rec}")
            else:
                st.success(f"• {rec}")
        
        st.markdown("---")
        
        # Explanation
        st.subheader("📝 הסבר")
        st.write(alert.explanation)
    
    # Technical details (expandable)
    with st.expander("🔧 פרטים טכניים"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Baseline", f"{results['baseline'].value:.0f} bpm")
            st.metric("שונות", f"{results['variability'].value:.1f} bpm")
        
        with col_b:
            st.metric("קטגוריית שונות", results['variability'].category.name)
            late_count = sum(1 for d in results['decelerations'] 
                           if d.decel_type == DecelerationType.LATE)
            st.metric("האטות מאוחרות", late_count)
        
        with col_c:
            variable_count = sum(1 for d in results['decelerations'] 
                               if d.decel_type == DecelerationType.VARIABLE)
            st.metric("האטות משתנות", variable_count)
            st.metric("סה״כ האטות", len(results['decelerations']))


def render_no_model_warning():
    """Render warning when model is not available."""
    st.warning(
        """
        ⚠️ **מודל ML לא נמצא**
        
        לא נמצא קובץ מודל ב-`models/xgb_demo.json`.
        המערכת תשתמש בסיווג מבוסס חוקים בלבד.
        
        לאימון מודל הרץ:
        ```
        python src/training/train_demo.py
        ```
        """
    )


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    # Page config
    st.set_page_config(
        page_title="SentinelFetal - מערכת ניטור עוברי",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title
    st.title("🏥 SentinelFetal - מערכת ניטור עוברי")
    st.markdown("מערכת AI לניטור דופק עוברי בזמן לידה")
    st.markdown("---")
    
    # Load data
    loader = load_data_loader()
    if loader is None:
        st.error("לא ניתן לטעון את מאגר הנתונים. בדוק את הנתיב.")
        st.stop()
    
    # Check for model
    classifier = load_classifier()
    if classifier is None:
        render_no_model_warning()
    
    # Sidebar - Patient selection
    selected_record = render_sidebar(loader)
    
    if selected_record:
        # Process selected record
        with st.spinner(f"מעבד רשומה {selected_record}..."):
            try:
                record = loader.load_record(selected_record)
                use_ml = st.session_state.get('use_ml', True)
                results = run_full_pipeline(record, use_ml=use_ml)
                
                # Render results
                render_patient_detail(results, selected_record)
                
            except Exception as e:
                st.error(f"שגיאה בעיבוד הרשומה: {e}")
                logger.exception("Pipeline error")
    else:
        st.info("בחר רשומה מהתפריט בצד שמאל")


if __name__ == "__main__":
    main()
