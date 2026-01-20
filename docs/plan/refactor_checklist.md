# SentinelFetal Refactoring Checklist

Based on: `compass_artifact_wf-65822bd7-ab9e-4a8d-935d-fb2353cd722e_text_markdown.md`

**Last Updated:** 2026-01-20

---

## Phase 1: Quick Logic Fixes (1-2 weeks)

### 1.1 Add Descent Time Feature (HIGHEST PRIORITY)
- [x] Create `src/analysis/deceleration_features.py` → Implemented in `src/rules/decelerations.py`
- [x] Implement `calculate_descent_time(fhr_signal, baseline)` function
  - Returns time from onset to nadir in seconds
- [x] Add classification logic:
  - `< 30 seconds` → Variable deceleration (abrupt)
  - `≥ 30 seconds` → Late/Early deceleration (gradual)
- [x] Implement fuzzy logic for borderline cases (25-35 seconds)
- [ ] Write unit tests for descent time calculation
- **Status:** ✅ COMPLETED - Expected impact: **20-30% accuracy improvement**

### 1.2 Add UC-Nadir Lag Feature
- [ ] Implement `calculate_uc_nadir_lag(fhr_signal, uc_signal)` function
- [ ] Calculate temporal lag between UC peak and deceleration nadir
- [ ] Add as feature to classifier when UC signal available
- [ ] Fallback to FHR-only morphological assessment when UC unavailable

### 1.3 Add Variability-Within-Deceleration Feature
- [ ] Implement `calculate_variability_in_decel(fhr_signal, dec_start, dec_end)`
- [ ] Reduced variability (<5 bpm) suggests Late deceleration
- [ ] Use for FHR-only Late deceleration detection

---

## Phase 2: Dependency Cleanup (1-2 days)

### 2.1 Update requirements.txt
- [x] Remove `momentfm` / `moment-foundation-model` (if present)
- [x] Remove heavy torch dependencies (if not needed elsewhere)
- [x] Add `sktime>=0.24.0` (MiniRocket implementation)
- [x] Add `tsai>=0.3.7` (optional: InceptionTime if deep learning needed) - commented out
- [x] Add `pywavelets>=1.4.0` (Coiflet 4 denoising)
- [x] Verify no breaking changes for existing code
- **Status:** ✅ COMPLETED

### 2.2 Clean Up Imports
- [x] Search codebase for MOMENT imports
- [x] Document files that need updating:
  - `src/models/moment_encoder.py` - kept for legacy compatibility
  - `src/adapters/model_adapters.py` - updated with MiniRocket support
  - Any pipeline files referencing MOMENT - updated adapters
- **Status:** ✅ COMPLETED - MiniRocket is now default, MOMENT is optional fallback

---

## Phase 3: Model Swap Implementation (2-4 weeks)

### 3.1 Implement MiniRocket Encoder
- [x] Create `src/models/minirocket_encoder.py`
- [x] Implement `MiniRocketEncoder` class with same interface as `MomentEncoder`
- [x] Key methods:
  ```python
  def fit(X_train)  # One-time fit (~10 min for large datasets)
  def transform(X)  # Transform signals to features
  def save(path)    # Save fitted parameters
  def load(path)    # Load fitted parameters
  ```
- [x] Use `sktime.transformations.panel.rocket.MiniRocket`
- **Status:** ✅ COMPLETED

### 3.2 Create Ridge Classifier Wrapper
- [x] Create lightweight classifier using `sklearn.linear_model.RidgeClassifierCV`
- [x] Expected inference: ~1ms per sample
- [x] Save/load model with joblib
- **Status:** ✅ COMPLETED (in MiniRocketEncoder)

### 3.3 Update Model Adapters
- [x] Modify `src/adapters/model_adapters.py` to support MiniRocket
- [x] Add config option to switch between MOMENT and MiniRocket
- [x] Default to MiniRocket for production
- [x] Add `get_feature_extractor()` factory function
- **Status:** ✅ COMPLETED

### 3.4 Preprocessing Improvements
- [x] Implement Coiflet 4 wavelet denoising in `src/data/signal_quality.py`
  ```python
  import pywt
  def denoise_coiflet4(signal, wavelet='coif4', level=5):
      # Soft thresholding + universal algorithm
  ```
- [x] Add FSQI (Fetal Signal Quality Index) gate
- [x] Create quality threshold: `if fsqi_score < 0.7: return "SIGNAL_QUALITY_INSUFFICIENT"`
- **Status:** ✅ COMPLETED

### 3.5 Retrain/Refit Model
- [ ] Fit MiniRocket on CTU-CHB training data
- [ ] Train Ridge classifier on transformed features
- [ ] Add new features (descent time, UC lag, etc.)
- [ ] Validate AUC target: **0.80-0.85**

### 3.6 INT8 Quantization (Optional)
- [ ] Quantize ONNX model if still using neural network components
- [ ] Expected: 2-4x speedup, 75% memory reduction
- [ ] Validate accuracy loss < 3%

---

## Phase 4: UI Optimization (1-2 weeks)

### 4.1 Implement Staggered Fragment Updates
- [x] Refactor patient tiles to use `@st.fragment`
- [x] Stagger refresh times: `run_every = f"{0.5 + (patient_id % 5) * 0.1}s"`
- **Status:** ✅ COMPLETED - Expected: **80%+ CPU reduction**

### 4.2 Optimize Chart Rendering
- [ ] Evaluate `streamlit-echarts` vs Plotly
- [ ] Implement LTTB downsampling for real-time charts
- [ ] Limit to last 50-100 points per chart
- [ ] Avoid Matplotlib (blocks widget interactions)

### 4.3 Caching Strategy
- [ ] Use `@st.cache_resource` for model loading (load once)
- [ ] Use `@st.cache_data(ttl=300)` for clinical thresholds
- [ ] Use Session State for real-time patient data (never cache)

### 4.4 Multi-Instance Inference Pool
- [ ] Implement `PatientInferencePool` for parallel processing
- [ ] Target: 4 instances on 8-core CPU
- [ ] Expected: 15-25ms for all 20 patients in parallel

---

## Phase 5: Validation & Testing

### 5.1 Unit Tests
- [ ] Test descent time calculation with known examples
- [ ] Test MiniRocket encoder interface
- [ ] Test preprocessing pipeline (Coiflet 4 + FSQI)

### 5.2 Integration Tests
- [ ] End-to-end pipeline with new model
- [ ] Verify sub-100ms inference time
- [ ] Validate deceleration classification accuracy

### 5.3 Performance Benchmarks
- [ ] Measure inference time per patient
- [ ] Measure total time for 20 concurrent patients
- [ ] Compare before/after: MOMENT vs MiniRocket

### 5.4 Clinical Validation
- [ ] Validate against Israeli Position Paper (FIGO 2015) definitions
- [ ] Target metrics:
  - Late deceleration sensitivity: >70% (currently 23%)
  - Variable deceleration sensitivity: >70% (currently 14%)
  - Overall AUC: 0.80-0.85

---

## Quick Reference: Expected Improvements

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Late Decel Detection | 23% | 70%+ | **3x** |
| Variable Decel Detection | 14% | 70%+ | **5x** |
| Inference Time | ~500ms+ | <100ms | **5-10x** |
| Model Size | 341M params | ~84 kernels | **~4000x smaller** |
| CPU Usage (UI) | High | -80% | **5x reduction** |

---

## Files Created/Modified (2026-01-20)

### New Files:
- `src/models/minirocket_encoder.py` ✅
- `src/data/signal_quality.py` ✅

### Files Modified:
- `requirements.txt` ✅
- `src/rules/decelerations.py` ✅ (30-second descent time rule)
- `src/adapters/model_adapters.py` ✅ (MiniRocket support)
- `src/models/__init__.py` ✅ (export new encoder)
- `src/ui/simulation_app.py` ✅ (staggered updates)

---

## Commands to Run

```bash
# After updating requirements.txt
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Benchmark inference
python benchmark_performance.py
```
