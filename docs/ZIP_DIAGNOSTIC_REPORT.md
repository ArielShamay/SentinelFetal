# ZIP_DIAGNOSTIC_REPORT (training_pack_v6_2.zip)

Generated: 2026-01-27 16:59:45

## A) ???? ZIP ??????
1) Path: C:/Users/ariel/OneDrive/שולחן העבודה/SentinelFetal/SentinelFetal/training_pack_v6_2.zip
   - Size: 96635284 bytes (92.16 MB)
   - Mtime: 2026-01-27 03:21:53
   - SHA256: 47670211D5DAEA1B6E4067DB1B9E86C333B04F5849B423D7FBA64C5BD7498F4D

2) Top-level entries (first 50):
- manifest.json (files: 1)
- records (files: 2081)

3) manifest.json:
- found at: manifest.json

## B) manifest.json ? ???? ?????
1) Root keys:
- build_timestamp, ctgdl_suggestions, datasets, git_commit, patients, quality_policy, runtime_config, stats

2) runtime_config (values + types):
```json
{
  "fs_hz": {
    "value": 4.0,
    "type": "float"
  },
  "window_minutes": {
    "value": 20.0,
    "type": "float"
  },
  "stride_minutes": {
    "value": 5.0,
    "type": "float"
  },
  "min_window_minutes": {
    "value": 20.0,
    "type": "float"
  }
}
```
- strict_mode present: False

3) manifest["patients"] statistics:
- total records: 3341

Source distribution
| key | count |
|---|---|
| CTGDL | 1532 |
| FHRMA | 1257 |
| CTU-CHB | 552 |

Included distribution
| key | count |
|---|---|
| True | 2768 |
| False | 573 |

Target_task distribution
| key | count |
|---|---|
| Anatomy | 1532 |
| Quality | 1257 |
| Outcome | 552 |

Label_type distribution
| key | count |
|---|---|
| none | 3006 |

Labels_present distribution
| key | count |
|---|---|
| False | 3006 |

Short_case distribution
| key | count |
|---|---|
| False | 2886 |
| True | 120 |

4) Patient object fields:
- avg_quality, duration_minutes, gap_fill, included, label_type, labels_present, patient_id, record_quality, reject_detail, short_case, skip_reason, source, target_task
- Possible path/file hint keys: record_quality

Sample patients (sanitized):
- CTU-CHB:
```json
{
  "patient_id": "1001",
  "source": "CTU-CHB",
  "duration_minutes": 80.0,
  "record_quality": {
    "fhr_valid_frac": 1.0,
    "fhr_nan_frac": 0.0,
    "fhr_zeros_frac": 0.22161458333333334,
    "fhr_out_of_range_frac": 0.22234375,
    "uc_valid_frac": 1.0,
    "uc_nan_frac": 0.0,
    "duration_minutes": 80.0,
    "flags": [
      "FHR_OUT_OF_RANGE_HIGH"
    ]
  },
  "labels_present": false,
  "label_type": "none",
  "target_task": "Outcome",
  "included": true,
  "skip_reason": null,
  "reject_detail": null,
  "short_case": false,
  "avg_quality": {
    "HIGH": 0.0,
    "MED": 0.8461538461538461,
    "LOW": 0.15384615384615385
  },
  "gap_fill": {
    "fhr": {
      "filled_gap_count": 0,
      "filled_samples": 0,
      "max_gap": 0,
      "total_nan_before": 0,
      "total_nan_after": 0
    },
    "uc": {
      "filled_gap_count": 0,
      "filled_samples": 0,
      "max_gap": 0,
      "total_nan_before": 0,
      "total_nan_after": 0
    },
    "policy": {
      "max_gap_samples_to_fill": 10,
      "fill_method": "linear",
      "do_not_fill_if_gap_too_large": true
    }
  }
}
```
- CTGDL:
```json
{
  "patient_id": "ctgdl_ctu_uhb_1001",
  "source": "CTGDL",
  "duration_minutes": 80.0,
  "record_quality": {
    "fhr_valid_frac": 1.0,
    "fhr_nan_frac": 0.0,
    "fhr_zeros_frac": 0.22161458333333334,
    "fhr_out_of_range_frac": 0.22234375,
    "uc_valid_frac": 1.0,
    "uc_nan_frac": 0.0,
    "duration_minutes": 80.0,
    "flags": [
      "FHR_OUT_OF_RANGE_HIGH"
    ]
  },
  "labels_present": false,
  "label_type": "none",
  "target_task": "Anatomy",
  "included": true,
  "skip_reason": null,
  "reject_detail": null,
  "short_case": false,
  "avg_quality": {
    "HIGH": 0.0,
    "MED": 0.8461538461538461,
    "LOW": 0.15384615384615385
  },
  "gap_fill": {
    "fhr": {
      "filled_gap_count": 0,
      "filled_samples": 0,
      "max_gap": 0,
      "total_nan_before": 0,
      "total_nan_after": 0
    },
    "uc": {
      "filled_gap_count": 0,
      "filled_samples": 0,
      "max_gap": 0,
      "total_nan_before": 0,
      "total_nan_after": 0
    },
    "policy": {
      "max_gap_samples_to_fill": 10,
      "fill_method": "linear",
      "do_not_fill_if_gap_too_large": true
    }
  }
}
```
- FHRMA:
```json
{
  "patient_id": "Example 10 Doppler Stage 2 at min 580",
  "source": "FHRMA",
  "duration_minutes": 599.7166666666667,
  "record_quality": {
    "fhr_valid_frac": 1.0,
    "fhr_nan_frac": 0.0,
    "fhr_zeros_frac": 0.10331962315537893,
    "fhr_out_of_range_frac": 0.10340994358447045,
    "uc_valid_frac": 1.0,
    "uc_nan_frac": 0.0,
    "duration_minutes": 599.7166666666667,
    "flags": []
  },
  "labels_present": false,
  "label_type": "none",
  "target_task": "Quality",
  "included": true,
  "skip_reason": null,
  "reject_detail": null,
  "short_case": false,
  "avg_quality": {
    "HIGH": 0.0,
    "MED": 0.9655172413793104,
    "LOW": 0.034482758620689655
  },
  "gap_fill": {
    "fhr": {
      "filled_gap_count": 0,
      "filled_samples": 0,
      "max_gap": 0,
      "total_nan_before": 0,
      "total_nan_after": 0
    },
    "uc": {
      "filled_gap_count": 0,
      "filled_samples": 0,
      "max_gap": 0,
      "total_nan_before": 0,
      "total_nan_after": 0
    },
    "policy": {
      "max_gap_samples_to_fill": 10,
      "fill_method": "linear",
      "do_not_fill_if_gap_too_large": true
    }
  }
}
```

Manifest filename/path hints check:
- Potential hint keys found (see list above); check values for redacted paths.

## C) ??????? ???? NPZ ???? ?-ZIP
1) Total .npz files: 2081
   - All under records/: True
   - Sample paths (first 20):
     - records/1001.npz
     - records/1002.npz
     - records/1003.npz
     - records/1004.npz
     - records/1005.npz
     - records/1006.npz
     - records/1007.npz
     - records/1008.npz
     - records/1009.npz
     - records/1010.npz
     - records/1011.npz
     - records/1012.npz
     - records/1013.npz
     - records/1014.npz
     - records/1015.npz
     - records/1016.npz
     - records/1017.npz
     - records/1018.npz
     - records/1019.npz
     - records/1020.npz

2) Mapping patient_id -> NPZ:
   - records/{patient_id}.npz matches: 2785 / 3341
   - any path ending /{patient_id}.npz matches: 2785 / 3341
   - missing: 556
   - collisions (same basename in multiple paths): 0

## D) ???? NPZ ? ????? ?????

### Patient 1001 (CTU-CHB)
- NPZ path: records/1001.npz
- keys: fhr_raw, uc_raw, fhr_filled, uc_filled, fs_hz, patient_id, source, labels_json, meta_json, record_quality_json
- fhr_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 169.25, 'mean': 142.55265, 'nan_pct': 0.0}
- fhr_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 169.25, 'mean': 142.55265, 'nan_pct': 0.0}
- uc_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 66.0, 'mean': 18.2995, 'nan_pct': 0.0}
- uc_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 66.0, 'mean': 18.2995, 'nan_pct': 0.0}
- fs_hz: dtype=<f8 shape=() value=4.0
- patient_id: dtype=<U4 shape=() value=1001
- source: dtype=<U7 shape=() value=CTU-CHB
- labels_json: dtype=<U2 shape=()
- meta_json: dtype=<U662 shape=()
- record_quality_json: dtype=<U224 shape=()
- fhr_filled is 1D: length=19200

### Patient 1002 (CTU-CHB)
- NPZ path: records/1002.npz
- keys: fhr_raw, uc_raw, fhr_filled, uc_filled, fs_hz, patient_id, source, labels_json, meta_json, record_quality_json
- fhr_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 182.0, 'mean': 149.53605, 'nan_pct': 0.0}
- fhr_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 182.0, 'mean': 149.53605, 'nan_pct': 0.0}
- uc_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 1.5, 'max': 105.0, 'mean': 30.7632, 'nan_pct': 0.0}
- uc_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 1.5, 'max': 105.0, 'mean': 30.7632, 'nan_pct': 0.0}
- fs_hz: dtype=<f8 shape=() value=4.0
- patient_id: dtype=<U4 shape=() value=1002
- source: dtype=<U7 shape=() value=CTU-CHB
- labels_json: dtype=<U2 shape=()
- meta_json: dtype=<U662 shape=()
- record_quality_json: dtype=<U189 shape=()
- fhr_filled is 1D: length=19200

### Patient ctgdl_ctu_uhb_1001 (CTGDL)
- NPZ path: records/ctgdl_ctu_uhb_1001.npz
- keys: fhr_raw, uc_raw, fhr_filled, uc_filled, fs_hz, patient_id, source, labels_json, meta_json, record_quality_json
- fhr_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 116.25, 'max': 169.25, 'mean': 143.623125, 'nan_pct': 0.0}
- fhr_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 116.25, 'max': 169.25, 'mean': 143.623125, 'nan_pct': 0.0}
- uc_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.5, 'max': 66.0, 'mean': 18.30875, 'nan_pct': 0.0}
- uc_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.5, 'max': 66.0, 'mean': 18.30875, 'nan_pct': 0.0}
- fs_hz: dtype=<f8 shape=() value=4.0
- patient_id: dtype=<U18 shape=() value=ctgdl_ctu_uhb_1001
- source: dtype=<U5 shape=() value=CTGDL
- labels_json: dtype=<U2 shape=()
- meta_json: dtype=<U700 shape=()
- record_quality_json: dtype=<U178 shape=()
- fhr_filled is 1D: length=19200

### Patient ctgdl_ctu_uhb_1002 (CTGDL)
- NPZ path: records/ctgdl_ctu_uhb_1002.npz
- keys: fhr_raw, uc_raw, fhr_filled, uc_filled, fs_hz, patient_id, source, labels_json, meta_json, record_quality_json
- fhr_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 135.75, 'max': 182.0, 'mean': 152.986075, 'nan_pct': 0.0}
- fhr_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 135.75, 'max': 182.0, 'mean': 152.986075, 'nan_pct': 0.0}
- uc_raw: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 1.5, 'max': 105.0, 'mean': 30.7632, 'nan_pct': 0.0}
- uc_filled: dtype=<f8 shape=(19200,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 1.5, 'max': 105.0, 'mean': 30.7632, 'nan_pct': 0.0}
- fs_hz: dtype=<f8 shape=() value=4.0
- patient_id: dtype=<U18 shape=() value=ctgdl_ctu_uhb_1002
- source: dtype=<U5 shape=() value=CTGDL
- labels_json: dtype=<U2 shape=()
- meta_json: dtype=<U700 shape=()
- record_quality_json: dtype=<U196 shape=()
- fhr_filled is 1D: length=19200

### Patient Example 10 Doppler Stage 2 at min 580 (FHRMA)
- NPZ path: records/Example 10 Doppler Stage 2 at min 580.npz
- keys: fhr_raw, uc_raw, fhr_filled, uc_filled, fs_hz, patient_id, source, labels_json, meta_json, record_quality_json
- fhr_raw: dtype=<f8 shape=(143932,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 164.75, 'mean': 130.29315, 'nan_pct': 0.0}
- fhr_filled: dtype=<f8 shape=(143932,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 164.75, 'mean': 130.29315, 'nan_pct': 0.0}
- uc_raw: dtype=<f8 shape=(143932,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 127.0, 'mean': 15.1852, 'nan_pct': 0.0}
- uc_filled: dtype=<f8 shape=(143932,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 127.0, 'mean': 15.1852, 'nan_pct': 0.0}
- fs_hz: dtype=<f8 shape=() value=4.0
- patient_id: dtype=<U37 shape=() value=Example 10 Doppler Stage 2 at min 580
- source: dtype=<U5 shape=() value=FHRMA
- labels_json: dtype=<U2 shape=()
- meta_json: dtype=<U915 shape=()
- record_quality_json: dtype=<U223 shape=()
- fhr_filled is 1D: length=143932

### Patient Example 2 Doppler Pattern D Stage 2 at min 275 (FHRMA)
- NPZ path: records/Example 2 Doppler Pattern D Stage 2 at min 275.npz
- keys: fhr_raw, uc_raw, fhr_filled, uc_filled, fs_hz, patient_id, source, labels_json, meta_json, record_quality_json
- fhr_raw: dtype=<f8 shape=(70009,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 167.75, 'mean': 121.3698, 'nan_pct': 0.0}
- fhr_filled: dtype=<f8 shape=(70009,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 0.0, 'max': 167.75, 'mean': 121.3698, 'nan_pct': 0.0}
- uc_raw: dtype=<f8 shape=(70009,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 10.0, 'max': 78.0, 'mean': 20.7518, 'nan_pct': 0.0}
- uc_filled: dtype=<f8 shape=(70009,) ndim=1 stats_sample(min/max/mean/nan%)={'min': 10.0, 'max': 78.0, 'mean': 20.7518, 'nan_pct': 0.0}
- fs_hz: dtype=<f8 shape=() value=4.0
- patient_id: dtype=<U46 shape=() value=Example 2 Doppler Pattern D Stage 2 at min 275
- source: dtype=<U5 shape=() value=FHRMA
- labels_json: dtype=<U2 shape=()
- meta_json: dtype=<U923 shape=()
- record_quality_json: dtype=<U224 shape=()
- fhr_filled is 1D: length=70009

## E) JSON Fields (labels/meta/record_quality)
### labels_json
- stored dtype=<U2 shape=()
- raw type: str
- raw snippet (redacted):
```text
{}
```
- json.loads success: True
- parsed type: dict
- keys (up to 100): 
- parsed empty dict: True
### meta_json
- stored dtype=<U662 shape=()
- raw type: str
- raw snippet (redacted):
```text
{"source_path": "<redacted_path>", "original_fs": 4.0, "selected_fhr_channel": "FHR", "selected_uc_channel": "UC", "sig_names": ["FHR", "UC"], "original_length": 19200, "resampled": false, "gap_fill": {"fhr": {"filled_gap_count": 0, "filled_samples": 0, "max_gap": 0, "total_nan_before": 0, "total_nan_after": 0}, "uc": {"filled_gap_count": 0, "filled_samples": 0, "max_gap": 0, "total_nan_before": 0, "total_nan_after": 0}, "policy": {"max_gap_samples_to_fill": 10, "fill_method": "linear", "do_not_fill_if_gap_too_large": true}}, "target_task": "Outcome"}
```
- json.loads success: True
- parsed type: dict
- keys (up to 100): gap_fill, original_fs, original_length, resampled, selected_fhr_channel, selected_uc_channel, sig_names, source_path, target_task
- nested keys (level 2, truncated):
```json
{
  "gap_fill": [
    "fhr",
    "policy",
    "uc"
  ]
}
```
- parsed empty dict: False
### record_quality_json
- stored dtype=<U224 shape=()
- raw type: str
- raw snippet (redacted):
```text
{"fhr_valid_frac": 1.0, "fhr_nan_frac": 0.0, "fhr_zeros_frac": 0.22161458333333334, "fhr_out_of_range_frac": 0.22234375, "uc_valid_frac": 1.0, "uc_nan_frac": 0.0, "duration_minutes": 80.0, "flags": ["FHR_OUT_OF_RANGE_HIGH"]}
```
- json.loads success: True
- parsed type: dict
- keys (up to 100): duration_minutes, fhr_nan_frac, fhr_out_of_range_frac, fhr_valid_frac, fhr_zeros_frac, flags, uc_nan_frac, uc_valid_frac
- parsed empty dict: False

labels_json empty check (sampled):
- empty dict count in samples: 6 / 6

Label key scan (from labels_json in up to 30 samples with labels_present=true):
- no label keys found in scanned labels_json samples

## F) ????? label ???? Outcome Model
- outcome_binary field present: no
- pH-like fields present: no
- manifest target_task indicates Outcome for some records: yes

Recipe suggestion (based on findings):
- If labels_json contains outcome/pH keys: parse labels_json and derive y from those keys.
- Else, use manifest fields: labels_present/label_type/target_task only indicate task, not label value.
- If no labels_json keys found for Outcome: you likely need to enrich labels from source metadata (CTU-CHB/CTGDL) before Step 1.

## G) Quality / Record Quality invariants
- record_quality_json keys (from sample): duration_minutes, fhr_nan_frac, fhr_out_of_range_frac, fhr_valid_frac, fhr_zeros_frac, flags, uc_nan_frac, uc_valid_frac
- Manifest fields include: included, skip_reason, reject_detail (per patient).
- If included=false or skip_reason/reject_detail present: skip record in downstream processing.

## H) Cross-check ??? runtime_config
- Outcome sample size: 10
- fs_hz match: 10/10 (100.0%)
- window length check: N/A (no 2D arrays in sampled Outcome records)

## I) ?????? ????
### ????? ??????
- NPZ files are stored under records/ and are named by patient_id (records/{patient_id}.npz).
- runtime_config in manifest does not include strict_mode.
- labels_json/meta_json/record_quality_json are stored as JSON strings in npz (parsed with json.loads).

### ?????? / ??-??????
- labels_json keys were not found in scanned samples; label values may be missing or encoded elsewhere.

### ??? ?????? ???? Step 1
- Open training_pack_v6_2.zip and read manifest.json from zip root.
- Build patient_id -> npz mapping using records/{patient_id}.npz.
- For each npz: load keys; use fhr_raw/uc_raw or fhr_filled/uc_filled as 1D signals.
- Use fs_hz from npz (validate against manifest runtime_config.fs_hz).
- Parse labels_json/meta_json/record_quality_json as JSON strings; handle empty dicts safely.
- Skip records where included=false or skip_reason/reject_detail present in manifest.
- If labels_json lacks outcome fields, Step 1 must integrate labels from source metadata (CTU-CHB/CTGDL) before training.