# SPECS - מפרט טכני מפורט
## שדרוג ביצועים SentinelFetal

**גרסה:** 1.0  
**תאריך:** 19 ינואר 2026  
**מבוסס על:** PRD v1.0  

---

## 1. ארכיטקטורה כללית

### 1.1 תרשים זרימה מעודכן

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Sidebar   │  │   Plotly    │  │      Alert Panel        │  │
│  │  (Patients) │  │   Graphs    │  │   (Hebrew XAI)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Pipeline Controller                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  @st.cache_resource: Models (MOMENT-OV, XGBoost)         │   │
│  │  @st.cache_data: Embeddings, Rule Results                │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Rule Engine  │  │  MOMENT-OpenVINO │  │    XGBoost     │
│  (DSP-based)  │  │  (INT8 Quantized)│  │   Classifier   │
│               │  │                  │  │                │
│  - Baseline   │  │  - 1024-dim      │  │  - 1035-dim    │
│  - Variability│  │    embeddings    │  │    input       │
│  - Decels     │  │  - < 500ms       │  │  - 3 classes   │
│  - Tachy      │  │  - ~400MB RAM    │  │                │
│  - Sinusoidal │  │                  │  │                │
└───────────────┘  └─────────────────┘  └─────────────────┘
```

### 1.2 תלויות (Dependencies)

```python
# requirements_optimized.txt

# Existing (keep)
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
xgboost>=2.0.0
wfdb>=4.1.0
streamlit>=1.28.0

# Updated (replace matplotlib)
plotly>=5.18.0          # Interactive charts

# New (optimization)
onnx>=1.14.0            # Model export format
onnxruntime>=1.15.0     # ONNX inference (fallback)
openvino>=2024.0.0      # Intel-optimized inference

# Remove (no longer needed for UI)
# matplotlib - replaced by plotly
```

---

## 2. שלב א' - שיפור UX

### 2.1 החלפת Matplotlib ב-Plotly

#### קובץ: `src/ui/plots.py`

**שינויים נדרשים:**

```python
"""
CTG Visualization - Plotly Version
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.rules.decelerations import Deceleration, DecelerationType
from src.config import COLORS


def create_ctg_plot(
    fhr: np.ndarray,
    uc: np.ndarray,
    decelerations: Optional[List[Deceleration]] = None,
    sampling_rate: float = 4.0,
    title: str = "CTG Monitor",
    height: int = 600
) -> go.Figure:
    """
    Create interactive CTG plot with Plotly.
    
    Key improvements over Matplotlib:
    - Zoom/Pan with mouse
    - Hover tooltips with exact values
    - Responsive to container width
    - No image rendering overhead
    
    Args:
        fhr: FHR signal array (bpm)
        uc: Uterine contractions array
        decelerations: List of detected decelerations to highlight
        sampling_rate: Hz (default 4)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object ready for st.plotly_chart()
    """
    # Time axis in minutes
    n_samples = len(fhr)
    time_minutes = np.arange(n_samples) / sampling_rate / 60
    
    # Create subplots: FHR (top 70%), UC (bottom 30%)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=('דופק עוברי (FHR)', 'צירים (UC)')
    )
    
    # FHR trace
    fig.add_trace(
        go.Scattergl(  # Scattergl for better performance with large datasets
            x=time_minutes,
            y=fhr,
            mode='lines',
            name='FHR',
            line=dict(color=COLORS.FHR, width=1.5),
            hovertemplate='זמן: %{x:.1f} דקות<br>דופק: %{y:.0f} bpm<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Normal range band (110-160 bpm)
    fig.add_hrect(
        y0=110, y1=160,
        fillcolor='rgba(40, 167, 69, 0.1)',
        line_width=0,
        row=1, col=1
    )
    
    # Deceleration highlights
    if decelerations:
        for decel in decelerations:
            start_min = decel.start_idx / sampling_rate / 60
            end_min = decel.end_idx / sampling_rate / 60
            
            # Color by severity
            if decel.decel_type == DecelerationType.LATE:
                color = 'rgba(220, 53, 69, 0.3)'  # Red - dangerous
            elif decel.decel_type == DecelerationType.VARIABLE and decel.has_severity_signs:
                color = 'rgba(255, 193, 7, 0.3)'  # Yellow - warning
            else:
                color = 'rgba(108, 117, 125, 0.2)'  # Gray - benign
            
            fig.add_vrect(
                x0=start_min, x1=end_min,
                fillcolor=color,
                line_width=0,
                row=1, col=1
            )
    
    # UC trace
    fig.add_trace(
        go.Scattergl(
            x=time_minutes,
            y=uc,
            mode='lines',
            name='UC',
            line=dict(color=COLORS.UC, width=1.5),
            hovertemplate='זמן: %{x:.1f} דקות<br>לחץ: %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Layout styling
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        height=height,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right'),
        hovermode='x unified',
        paper_bgcolor='#FAFAFA',
        plot_bgcolor='white',
        # Enable range slider for time navigation
        xaxis2=dict(rangeslider=dict(visible=True, thickness=0.05))
    )
    
    # Y-axis configuration
    fig.update_yaxes(title_text='bpm', range=[50, 220], dtick=30, 
                     gridcolor='#E5E5E5', row=1, col=1)
    fig.update_yaxes(title_text='mmHg', gridcolor='#E5E5E5', row=2, col=1)
    
    # X-axis
    fig.update_xaxes(title_text='זמן (דקות)', gridcolor='#E5E5E5', row=2, col=1)
    
    # Config for interactivity
    fig.update_layout(
        dragmode='zoom',
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def create_mini_trend(
    values: List[float],
    title: str,
    color: str = '#1E90FF'
) -> go.Figure:
    """
    Create a small sparkline-style trend chart.
    
    Useful for showing variability or baseline trend over time.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=100,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig
```

### 2.2 Loading States

#### קובץ: `src/ui/app.py` - עדכונים

```python
# הוסף בתחילת הקובץ
import time

# פונקציית עזר לחיווי התקדמות
def show_progress(steps: list, current_step: int, message: str):
    """Display progress bar during analysis."""
    progress = (current_step + 1) / len(steps)
    st.progress(progress, text=f"📊 {message}")


# עדכון הפונקציה run_full_pipeline
def run_full_pipeline_with_progress(record: CTGRecord, use_ml: bool = True) -> Dict[str, Any]:
    """
    Run pipeline with visual progress updates.
    """
    results = {}
    
    steps = ['preprocessing', 'rules', 'embeddings', 'classification', 'alert']
    progress_placeholder = st.empty()
    
    # Step 1: Preprocessing
    with progress_placeholder.container():
        show_progress(steps, 0, "מעבד סיגנל...")
    
    preprocessor = get_preprocessor()
    fhr_result = preprocessor.process(record.fhr)
    uc_result = preprocessor.process(record.uc)
    results['fhr'] = fhr_result.processed_signal
    results['uc'] = uc_result.processed_signal
    
    # Step 2: Rule Engine
    with progress_placeholder.container():
        show_progress(steps, 1, "מריץ מנוע חוקים...")
    
    results['baseline'] = calculate_baseline(results['fhr'])
    results['variability'] = calculate_variability(results['fhr'])
    results['decelerations'] = detect_decelerations(
        results['fhr'], results['uc'], results['baseline'].value
    )
    results['tachysystole'] = detect_tachysystole(results['uc'])
    results['sinusoidal'] = detect_sinusoidal_pattern(results['fhr'])
    
    if use_ml:
        # Step 3: MOMENT Embeddings
        with progress_placeholder.container():
            show_progress(steps, 2, "מחשב embeddings...")
        
        moment = get_moment_encoder()  # Cached
        embeddings = extract_embeddings_sliding_window(
            fhr=results['fhr'],
            extractor=moment,
            window_minutes=10,
            step_minutes=1
        )
        
        # Step 4: Classification
        with progress_placeholder.container():
            show_progress(steps, 3, "מסווג...")
        
        classifier = get_classifier()
        if classifier and embeddings:
            # Build features and predict
            feature_vectors = [
                build_feature_vector(
                    embedding=emb.embedding,
                    baseline=results['baseline'],
                    variability=results['variability'],
                    decelerations=results['decelerations'],
                    tachysystole=results['tachysystole'],
                    sinusoidal=results['sinusoidal']
                )
                for emb in embeddings
            ]
            
            X = build_feature_matrix(feature_vectors)
            predictions = classifier.predict(X)
            probs = classifier.predict_proba(X)
            
            # Majority vote
            from collections import Counter
            results['ml_prediction'] = Counter(predictions).most_common(1)[0][0]
            results['confidence'] = float(np.mean(np.max(probs, axis=1)))
        else:
            results['ml_prediction'] = _rule_based_classify(
                results['variability'], results['decelerations'],
                results['baseline'], results['sinusoidal']
            )
            results['confidence'] = 0.7
    else:
        results['ml_prediction'] = _rule_based_classify(
            results['variability'], results['decelerations'],
            results['baseline'], results['sinusoidal']
        )
        results['confidence'] = 0.7
    
    # Step 5: Generate Alert
    with progress_placeholder.container():
        show_progress(steps, 4, "מייצר דו\"ח...")
    
    override = apply_medical_override(
        ml_prediction=results['ml_prediction'],
        baseline=results['baseline'],
        variability=results['variability'],
        decelerations=results['decelerations'],
        tachysystole=results['tachysystole'],
        sinusoidal=results['sinusoidal']
    )
    
    results['final_category'] = override.final_category + 1
    results['override_result'] = override
    
    results['alert'] = generate_alert(
        category=results['final_category'],
        confidence=results['confidence'],
        baseline=results['baseline'],
        variability=results['variability'],
        decelerations=results['decelerations'],
        tachysystole=results['tachysystole'],
        sinusoidal=results['sinusoidal']
    )
    
    # Clear progress
    progress_placeholder.empty()
    
    return results
```

### 2.3 Caching Strategy

#### קובץ: `src/ui/app.py` - Cache Decorators

```python
# Model caching - loaded once per session
@st.cache_resource(show_spinner=False)
def get_data_loader():
    """Load data loader (singleton)."""
    return CTUDataLoader(DATA_DIR)


@st.cache_resource(show_spinner="טוען מודל MOMENT...")
def get_moment_encoder():
    """
    Load MOMENT encoder with OpenVINO optimization.
    
    This is a heavy operation (~2-5 seconds) but only happens once.
    """
    from src.models.moment_onnx import MomentOpenVINO
    return MomentOpenVINO()  # New optimized class


@st.cache_resource(show_spinner=False)
def get_classifier():
    """Load XGBoost classifier."""
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        classifier = XGBClassifierWrapper()
        classifier.load_model(str(model_path))
        return classifier
    return None


# Data caching - per record
@st.cache_data(show_spinner=False, ttl=3600)  # 1 hour TTL
def get_preprocessed_record(_loader: CTUDataLoader, record_id: str):
    """
    Load and preprocess a record.
    
    Cached by record_id - same record won't be reloaded.
    """
    record = _loader.load_record(record_id)
    preprocessor = CTGPreprocessor()
    
    fhr_result = preprocessor.process(record.fhr)
    uc_result = preprocessor.process(record.uc)
    
    return {
        'record': record,
        'fhr': fhr_result.processed_signal,
        'uc': uc_result.processed_signal,
        'fhr_stats': fhr_result.stats,
        'uc_stats': uc_result.stats
    }


@st.cache_data(show_spinner=False, ttl=3600)
def get_embeddings(_encoder, fhr: np.ndarray, record_id: str) -> List:
    """
    Cache MOMENT embeddings per record.
    
    This is the expensive operation - caching saves ~2-5 seconds per record.
    """
    return extract_embeddings_sliding_window(
        fhr=fhr,
        extractor=_encoder,
        window_minutes=10,
        step_minutes=1
    )
```

---

## 3. שלב ב' - אופטימיזציית מודל

### 3.1 ייצוא MOMENT ל-ONNX

#### קובץ חדש: `scripts/export_moment_onnx.py`

```python
"""
Export MOMENT model to ONNX format.

Usage:
    python scripts/export_moment_onnx.py --output models/moment.onnx
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
except ImportError:
    MOMENT_AVAILABLE = False
    print("momentfm not installed. Cannot export.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_moment_to_onnx(output_path: str, opset_version: int = 17):
    """
    Export MOMENT-1-large to ONNX format.
    
    Args:
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (17 recommended for Transformers)
    """
    if not MOMENT_AVAILABLE:
        raise ImportError("momentfm package required")
    
    logger.info("Loading MOMENT-1-large model...")
    
    # Load model in embedding mode
    pipeline = MOMENTPipeline.from_pretrained(
        'AutonLab/MOMENT-1-large',
        model_kwargs={
            'task_name': 'embedding',
            'n_channels': 1
        }
    )
    
    model = pipeline.model
    model.eval()
    
    # Create dummy input: [batch=1, channels=1, seq_len=2400]
    # 2400 = 10 minutes * 60 seconds * 4 Hz
    dummy_input = torch.randn(1, 1, 2400, dtype=torch.float32)
    
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    
    # Export with dynamic axes for flexible sequence length
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['fhr_input'],
        output_names=['embedding'],
        dynamic_axes={
            'fhr_input': {0: 'batch_size', 2: 'seq_len'},
            'embedding': {0: 'batch_size'}
        },
        do_constant_folding=True,
        export_params=True
    )
    
    logger.info(f"ONNX model saved to: {output_path}")
    
    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed!")
    
    # Print model size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {file_size_mb:.1f} MB")
    
    return output_path


def verify_onnx_output(onnx_path: str, pytorch_model=None):
    """
    Verify ONNX output matches PyTorch output.
    """
    import onnxruntime as ort
    
    # Create test input
    test_input = np.random.randn(1, 1, 2400).astype(np.float32)
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {'fhr_input': test_input})[0]
    
    logger.info(f"ONNX output shape: {onnx_output.shape}")
    logger.info(f"ONNX output sample: {onnx_output[0, :5]}")
    
    # Compare with PyTorch if model provided
    if pytorch_model is not None:
        with torch.no_grad():
            pt_input = torch.tensor(test_input)
            pt_output = pytorch_model(pt_input)
            pt_output = pt_output.embeddings.numpy()
        
        diff = np.abs(onnx_output - pt_output).max()
        logger.info(f"Max difference from PyTorch: {diff:.2e}")
        
        if diff < 1e-4:
            logger.info("✓ ONNX output matches PyTorch!")
        else:
            logger.warning(f"⚠ Output difference: {diff:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="models/moment.onnx")
    args = parser.parse_args()
    
    export_moment_to_onnx(args.output)
```

### 3.2 Quantization ל-INT8

#### קובץ חדש: `scripts/quantize_moment.py`

```python
"""
Quantize MOMENT ONNX model to INT8.

Usage:
    python scripts/quantize_moment.py --input models/moment.onnx --output models/moment_int8.onnx
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quantize_model(input_path: str, output_path: str):
    """
    Apply dynamic INT8 quantization to ONNX model.
    
    Dynamic quantization is best for Transformer models because:
    - Activations are quantized at runtime (no calibration dataset needed)
    - Weights are quantized statically
    - Good balance between speed and accuracy
    
    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path to save INT8 model
    """
    logger.info(f"Loading model from: {input_path}")
    
    # Dynamic quantization for Transformers
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,  # INT8 weights
        optimize_model=True,
        extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True
        }
    )
    
    # Compare sizes
    original_size = Path(input_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    logger.info(f"Original model size: {original_size:.1f} MB")
    logger.info(f"Quantized model size: {quantized_size:.1f} MB")
    logger.info(f"Compression ratio: {original_size / quantized_size:.2f}x")


def benchmark_models(fp32_path: str, int8_path: str, n_iterations: int = 10):
    """
    Benchmark FP32 vs INT8 inference speed.
    """
    import onnxruntime as ort
    import time
    
    # Prepare test input
    test_input = np.random.randn(1, 1, 2400).astype(np.float32)
    
    # FP32 benchmark
    session_fp32 = ort.InferenceSession(fp32_path)
    
    # Warmup
    for _ in range(3):
        session_fp32.run(None, {'fhr_input': test_input})
    
    start = time.time()
    for _ in range(n_iterations):
        session_fp32.run(None, {'fhr_input': test_input})
    fp32_time = (time.time() - start) / n_iterations * 1000
    
    # INT8 benchmark
    session_int8 = ort.InferenceSession(int8_path)
    
    # Warmup
    for _ in range(3):
        session_int8.run(None, {'fhr_input': test_input})
    
    start = time.time()
    for _ in range(n_iterations):
        session_int8.run(None, {'fhr_input': test_input})
    int8_time = (time.time() - start) / n_iterations * 1000
    
    logger.info(f"FP32 inference time: {fp32_time:.1f} ms")
    logger.info(f"INT8 inference time: {int8_time:.1f} ms")
    logger.info(f"Speedup: {fp32_time / int8_time:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="models/moment.onnx")
    parser.add_argument("--output", type=str, default="models/moment_int8.onnx")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    
    quantize_model(args.input, args.output)
    
    if args.benchmark:
        benchmark_models(args.input, args.output)
```

### 3.3 OpenVINO Integration

#### קובץ חדש: `src/models/moment_onnx.py`

```python
"""
MOMENT Feature Extractor with OpenVINO/ONNX Runtime optimization.

This module replaces the original PyTorch-based MOMENT encoder with
an optimized version using OpenVINO for Intel CPUs or ONNX Runtime as fallback.

Performance targets:
- Inference time: < 500ms on Intel i5
- Memory usage: < 500MB
- Accuracy: > 98% of original FP32

Usage:
    >>> encoder = MomentOpenVINO()  # Loads optimized model
    >>> embedding = encoder.extract(fhr_window)  # 1024-dim vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)

# Check available backends
OPENVINO_AVAILABLE = False
ONNX_AVAILABLE = False

try:
    from openvino import Core, CompiledModel
    OPENVINO_AVAILABLE = True
    logger.info("OpenVINO available - will use Intel-optimized inference")
except ImportError:
    logger.info("OpenVINO not available, checking for ONNX Runtime...")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("ONNX Runtime available")
except ImportError:
    logger.warning("Neither OpenVINO nor ONNX Runtime available!")


@dataclass
class MomentConfig:
    """Configuration for MOMENT model."""
    model_path_onnx: str = "models/moment_int8.onnx"
    model_path_ov: str = "models/moment_int8_ov"  # OpenVINO IR directory
    embedding_dim: int = 1024
    default_window_samples: int = 2400  # 10 min * 60 sec * 4 Hz
    device: str = "CPU"  # OpenVINO device


class MomentOpenVINO:
    """
    Optimized MOMENT encoder using OpenVINO or ONNX Runtime.
    
    This class provides the same interface as the original MomentFeatureExtractor
    but uses optimized inference backends for Intel CPUs.
    
    Backend priority:
    1. OpenVINO (best performance on Intel)
    2. ONNX Runtime (cross-platform fallback)
    3. Mock mode (for testing without model)
    
    Attributes:
        EMBEDDING_DIM: Output dimension (1024)
        backend: Active backend ('openvino', 'onnxruntime', or 'mock')
    """
    
    EMBEDDING_DIM = 1024
    
    def __init__(self, config: Optional[MomentConfig] = None):
        """
        Initialize the optimized MOMENT encoder.
        
        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or MomentConfig()
        self._model = None
        self._session = None
        self.backend = 'mock'
        
        self._load_model()
    
    def _load_model(self):
        """Load model using best available backend."""
        
        # Try OpenVINO first (best for Intel)
        if OPENVINO_AVAILABLE:
            ov_path = Path(self.config.model_path_ov)
            if (ov_path / "moment.xml").exists():
                try:
                    self._load_openvino(ov_path)
                    return
                except Exception as e:
                    logger.warning(f"OpenVINO load failed: {e}")
        
        # Try ONNX Runtime
        if ONNX_AVAILABLE:
            onnx_path = Path(self.config.model_path_onnx)
            if onnx_path.exists():
                try:
                    self._load_onnx(onnx_path)
                    return
                except Exception as e:
                    logger.warning(f"ONNX load failed: {e}")
        
        # Fallback to mock
        logger.warning("No model found - using mock mode")
        self.backend = 'mock'
    
    def _load_openvino(self, model_dir: Path):
        """Load OpenVINO IR model."""
        logger.info(f"Loading OpenVINO model from {model_dir}")
        
        core = Core()
        
        # Read model
        model = core.read_model(str(model_dir / "moment.xml"))
        
        # Compile for CPU with optimizations
        self._model = core.compile_model(
            model,
            device_name=self.config.device,
            config={
                'PERFORMANCE_HINT': 'LATENCY',  # Optimize for single inference
                'NUM_STREAMS': '1',
                'INFERENCE_PRECISION_HINT': 'f32'  # Use FP32 for accuracy
            }
        )
        
        self.backend = 'openvino'
        logger.info("OpenVINO model loaded successfully")
    
    def _load_onnx(self, model_path: Path):
        """Load ONNX Runtime session."""
        logger.info(f"Loading ONNX model from {model_path}")
        
        # Configure session options for CPU
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.backend = 'onnxruntime'
        logger.info("ONNX Runtime session created successfully")
    
    def extract(self, fhr: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract 1024-dim embedding from FHR signal.
        
        Args:
            fhr: FHR signal array (1D, values in bpm)
            normalize: Whether to normalize input (zero mean, unit std)
            
        Returns:
            1024-dimensional embedding vector
        """
        # Prepare input
        fhr_prepared = self._prepare_input(fhr, normalize)
        
        # Run inference based on backend
        if self.backend == 'openvino':
            return self._extract_openvino(fhr_prepared)
        elif self.backend == 'onnxruntime':
            return self._extract_onnx(fhr_prepared)
        else:
            return self._extract_mock(fhr_prepared)
    
    def _prepare_input(self, fhr: np.ndarray, normalize: bool) -> np.ndarray:
        """
        Prepare FHR signal for model input.
        
        - Pad/truncate to window size
        - Replace NaN with zeros
        - Normalize if requested
        - Reshape to [1, 1, seq_len]
        """
        window_size = self.config.default_window_samples
        
        # Handle length
        if len(fhr) < window_size:
            # Pad with mean
            valid = fhr[~np.isnan(fhr)]
            pad_val = float(np.mean(valid)) if len(valid) > 0 else 0.0
            fhr = np.pad(fhr, (0, window_size - len(fhr)), 
                        constant_values=pad_val)
        elif len(fhr) > window_size:
            fhr = fhr[:window_size]
        
        # Replace NaN
        fhr = np.nan_to_num(fhr, nan=0.0)
        
        # Normalize
        if normalize:
            mean = np.mean(fhr)
            std = np.std(fhr)
            if std > 1e-8:
                fhr = (fhr - mean) / std
            else:
                fhr = fhr - mean
        
        # Reshape: [1, 1, seq_len]
        return fhr.reshape(1, 1, -1).astype(np.float32)
    
    def _extract_openvino(self, fhr: np.ndarray) -> np.ndarray:
        """Extract embedding using OpenVINO."""
        result = self._model([fhr])
        return result[0].flatten()
    
    def _extract_onnx(self, fhr: np.ndarray) -> np.ndarray:
        """Extract embedding using ONNX Runtime."""
        output = self._session.run(None, {'fhr_input': fhr})
        return output[0].flatten()
    
    def _extract_mock(self, fhr: np.ndarray) -> np.ndarray:
        """Generate deterministic mock embedding based on signal statistics."""
        # Use signal hash for reproducibility
        signal_hash = hash(fhr.tobytes()) % (2**32)
        rng = np.random.RandomState(signal_hash)
        
        embedding = rng.randn(self.EMBEDDING_DIM).astype(np.float32)
        
        # Encode signal statistics
        flat = fhr.flatten()
        embedding[0] = float(np.mean(flat))
        embedding[1] = float(np.std(flat))
        embedding[2] = float(np.max(flat) - np.min(flat))
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        
        return embedding
    
    def is_available(self) -> bool:
        """Check if real model (not mock) is available."""
        return self.backend != 'mock'
    
    def get_backend_info(self) -> dict:
        """Get information about the active backend."""
        return {
            'backend': self.backend,
            'embedding_dim': self.EMBEDDING_DIM,
            'window_samples': self.config.default_window_samples,
            'model_path': (
                self.config.model_path_ov if self.backend == 'openvino'
                else self.config.model_path_onnx
            )
        }


def convert_onnx_to_openvino(onnx_path: str, output_dir: str):
    """
    Convert ONNX model to OpenVINO IR format.
    
    This should be run once after ONNX export.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save OpenVINO IR files
    """
    from openvino import Core, save_model
    
    logger.info(f"Converting {onnx_path} to OpenVINO IR...")
    
    core = Core()
    model = core.read_model(onnx_path)
    
    # Save as IR
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_model(model, str(output_path / "moment.xml"))
    
    logger.info(f"OpenVINO IR saved to {output_dir}")


# Convenience function matching original API
def extract_embeddings_sliding_window(
    fhr: np.ndarray,
    extractor: MomentOpenVINO,
    sampling_rate: float = 4.0,
    window_minutes: float = 10.0,
    step_minutes: float = 1.0
) -> List[dict]:
    """
    Extract embeddings using sliding window.
    
    Same interface as original moment_encoder.extract_embeddings_sliding_window()
    but uses the optimized encoder.
    """
    window_samples = int(window_minutes * 60 * sampling_rate)
    step_samples = int(step_minutes * 60 * sampling_rate)
    
    results = []
    n_samples = len(fhr)
    
    if n_samples < window_samples:
        # Single window for short signals
        embedding = extractor.extract(fhr)
        results.append({
            'embedding': embedding,
            'start_idx': 0,
            'end_idx': n_samples,
            'start_time_sec': 0.0,
            'end_time_sec': n_samples / sampling_rate
        })
        return results
    
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = fhr[start:end]
        
        embedding = extractor.extract(window)
        
        results.append({
            'embedding': embedding,
            'start_idx': start,
            'end_idx': end,
            'start_time_sec': start / sampling_rate,
            'end_time_sec': end / sampling_rate
        })
    
    return results
```

---

## 4. בדיקות

### 4.1 בדיקות יחידה

#### קובץ: `tests/test_moment_onnx.py`

```python
"""
Tests for optimized MOMENT encoder.
"""

import pytest
import numpy as np

from src.models.moment_onnx import MomentOpenVINO, MomentConfig


class TestMomentOpenVINO:
    """Test suite for MomentOpenVINO."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return MomentOpenVINO()
    
    @pytest.fixture
    def sample_fhr(self):
        """Generate sample FHR signal (10 minutes @ 4Hz)."""
        # Simulate realistic FHR: 140 bpm ± 10 bpm
        np.random.seed(42)
        return 140 + 10 * np.random.randn(2400)
    
    def test_embedding_dimension(self, encoder, sample_fhr):
        """Embedding should be 1024-dimensional."""
        embedding = encoder.extract(sample_fhr)
        assert embedding.shape == (1024,)
    
    def test_embedding_deterministic(self, encoder, sample_fhr):
        """Same input should produce same output."""
        emb1 = encoder.extract(sample_fhr)
        emb2 = encoder.extract(sample_fhr)
        np.testing.assert_array_almost_equal(emb1, emb2)
    
    def test_handles_nan(self, encoder, sample_fhr):
        """Should handle NaN values gracefully."""
        fhr_with_nan = sample_fhr.copy()
        fhr_with_nan[100:200] = np.nan  # Gap
        
        embedding = encoder.extract(fhr_with_nan)
        assert not np.any(np.isnan(embedding))
    
    def test_short_signal(self, encoder):
        """Should handle signals shorter than window."""
        short_fhr = np.random.randn(1000)  # Only ~4 minutes
        
        embedding = encoder.extract(short_fhr)
        assert embedding.shape == (1024,)
    
    def test_inference_time(self, encoder, sample_fhr):
        """Inference should be < 500ms."""
        import time
        
        # Warmup
        for _ in range(3):
            encoder.extract(sample_fhr)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            encoder.extract(sample_fhr)
            times.append(time.time() - start)
        
        avg_time_ms = np.mean(times) * 1000
        assert avg_time_ms < 500, f"Inference too slow: {avg_time_ms:.1f}ms"


class TestPlotlyIntegration:
    """Test Plotly chart creation."""
    
    def test_ctg_plot_creation(self):
        """CTG plot should be created without errors."""
        from src.ui.plots import create_ctg_plot
        
        fhr = 140 + 10 * np.random.randn(2400)
        uc = 20 + 5 * np.random.randn(2400)
        
        fig = create_ctg_plot(fhr, uc)
        
        assert fig is not None
        assert len(fig.data) >= 2  # FHR and UC traces
```

### 4.2 בדיקת אינטגרציה

```python
# tests/test_integration.py

def test_full_pipeline_performance():
    """
    Integration test: full pipeline should complete in < 5 seconds.
    """
    import time
    from src.data.loader import CTUDataLoader
    from src.ui.app import run_full_pipeline
    
    loader = CTUDataLoader("data/...")
    record = loader.load_record("1001")
    
    start = time.time()
    results = run_full_pipeline(record, use_ml=True)
    elapsed = time.time() - start
    
    assert elapsed < 5.0, f"Pipeline too slow: {elapsed:.1f}s"
    assert 'alert' in results
    assert results['final_category'] in [1, 2, 3]
```

---

## 5. הוראות התקנה

### 5.1 התקנת תלויות חדשות

```bash
# Install new dependencies
pip install plotly>=5.18.0
pip install onnx>=1.14.0
pip install onnxruntime>=1.15.0
pip install openvino>=2024.0.0

# Or use requirements file
pip install -r requirements_optimized.txt
```

### 5.2 הכנת מודל מותאם

```bash
# Step 1: Export to ONNX (requires momentfm installed)
python scripts/export_moment_onnx.py --output models/moment.onnx

# Step 2: Quantize to INT8
python scripts/quantize_moment.py \
    --input models/moment.onnx \
    --output models/moment_int8.onnx \
    --benchmark

# Step 3: Convert to OpenVINO (optional, for best Intel performance)
python -c "
from src.models.moment_onnx import convert_onnx_to_openvino
convert_onnx_to_openvino('models/moment_int8.onnx', 'models/moment_int8_ov')
"
```

### 5.3 הפעלה

```bash
# Run the optimized dashboard
streamlit run src/ui/app.py
```

---

## 6. רשימת משימות ליישום

### Phase 1 - UX (Days 1-3)
- [ ] Update `src/ui/plots.py` - replace matplotlib with plotly
- [ ] Update `src/ui/app.py` - add caching decorators
- [ ] Update `src/ui/app.py` - add progress indicators
- [ ] Test all UI interactions
- [ ] Verify plotly charts work in Streamlit

### Phase 2 - Model Optimization (Days 4-7)
- [ ] Create `scripts/export_moment_onnx.py`
- [ ] Create `scripts/quantize_moment.py`
- [ ] Create `src/models/moment_onnx.py`
- [ ] Export MOMENT to ONNX
- [ ] Quantize to INT8
- [ ] Convert to OpenVINO IR
- [ ] Benchmark: verify < 500ms inference
- [ ] Integration test: verify accuracy preserved

### Phase 3 - Integration (Days 8-10)
- [ ] Update `src/ui/app.py` to use `MomentOpenVINO`
- [ ] Implement embedding caching
- [ ] Add lazy loading
- [ ] Full regression testing
- [ ] Update documentation
- [ ] Performance validation report

---

**סוף מסמך SPECS**
