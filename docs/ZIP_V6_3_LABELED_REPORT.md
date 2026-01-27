# ZIP V6.3 Labeled Report

**מטרה:** pack חדש שמכיל לייבלי Outcome אמיתיים עבור CTU-CHB (pH מה־.hea) ומסמן חסרים ב־CTGDL.

## סיכום קבצים
- zip: `training_pack_v6_3_labeled.zip`
- גודל: 96735524 bytes
- SHA256: `1626a59f5889fcbadd9c278e5c7d44fc4a962919bf323f79410819b5e355cf91`
- labels_table: `data/labels/labels_table.csv` (SHA256 `07b1f6b5308d54a3711112fd2ceaa747629cfe0894f30a85482e8362f3a65c92`)

## runtime_config
```json
{
  "fs_hz": 4.0,
  "window_minutes": 20.0,
  "stride_minutes": 5.0,
  "min_window_minutes": 20.0
}
```

## מקור לייבלים וכלל בינארי
- מקור: CTU-CHB .hea (שדות pH/BE/BDecf/Apgar).
- כלל outcome_binary: `ph < 7.15` -> 1, אחרת 0.
- CTGDL: לא נמצאו לייבלי Outcome בקבצים המקומיים -> מסומן missing_label.

## manifest.json
- מפתחות root: build_timestamp, git_commit, runtime_config, quality_policy, datasets, patients, ctgdl_suggestions, stats, label_pack

**דוגמת רשומה לכל מקור (patient):**
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
  "labels_present": true,
  "label_type": "ph",
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
  "target_task": "Outcome",
  "included": false,
  "skip_reason": "missing_label",
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
  "patient_id": "defaultpos",
  "source": "FHRMA",
  "target_task": "Quality",
  "included": false,
  "skip_reason": "MAT_KEYS_UNKNOWN"
}
```

## מבנה NPZ (דוגמאות)
- patient_id `1001` (source CTU-CHB):
```text
fhr_raw: descr=<f8 shape=[19200]
uc_raw: descr=<f8 shape=[19200]
fhr_filled: descr=<f8 shape=[19200]
uc_filled: descr=<f8 shape=[19200]
fs_hz: descr=<f8 shape=[]
patient_id: descr=<U4 shape=[]
source: descr=<U7 shape=[]
meta_json: descr=<U662 shape=[]
record_quality_json: descr=<U224 shape=[]
labels_json: descr=<U190 shape=[]
```
- patient_id `ctgdl_ctu_uhb_1001` (source CTGDL):
```text
fhr_raw: descr=<f8 shape=[19200]
uc_raw: descr=<f8 shape=[19200]
fhr_filled: descr=<f8 shape=[19200]
uc_filled: descr=<f8 shape=[19200]
fs_hz: descr=<f8 shape=[]
patient_id: descr=<U18 shape=[]
source: descr=<U5 shape=[]
labels_json: descr=<U2 shape=[]
meta_json: descr=<U700 shape=[]
record_quality_json: descr=<U178 shape=[]
```
- patient_id `1002` (source CTU-CHB):
```text
fhr_raw: descr=<f8 shape=[19200]
uc_raw: descr=<f8 shape=[19200]
fhr_filled: descr=<f8 shape=[19200]
uc_filled: descr=<f8 shape=[19200]
fs_hz: descr=<f8 shape=[]
patient_id: descr=<U4 shape=[]
source: descr=<U7 shape=[]
meta_json: descr=<U662 shape=[]
record_quality_json: descr=<U189 shape=[]
labels_json: descr=<U189 shape=[]
```

## labels_json (דוגמה מקוצרת)
```json
{"outcome_binary":1,"ph":7.14,"label_version":"v1_ph<7.15","label_source":"CTU-CHB .hea","label_origin_file":"1001.hea","target_task":"Outcome","be":-10.5,"bdecf":8.14,"apgar1":6,"apgar5":8}
```

## סטטיסטיקות תיוג
- CTU-CHB עם pH: 552 מתוך 552
- CTU-CHB חסר pH: 0
- CTGDL total: 1532 (כל הרשומות missing_label)
- pH min/mean/max: 6.850 / 7.230 / 7.470
- class balance (CTU-CHB, לפי ph<7.15): 0=447 / 1=105

## ולידציה (מדגם 30 NPZ)
- labels_json_missing: 0
- labels_json_empty: 13
- outcome_binary_invalid: 0
- ph_out_of_range: 0
- fhr_shape_issues: 0

## Contract ל-Step 1/2
- לקרוא `manifest.json` ולסנן `target_task == "Outcome"` ו-`included == true`.
- לכל NPZ: לצפות ל-`labels_json` עם `outcome_binary` ו-`ph` (CTU-CHB).
- אם `labels_present == false` או `skip_reason == missing_label` -> לדלג.
- מבנה signals קיים: `fhr_filled` ו-`uc_filled`, בד״כ shape (n_windows, 4800) או רציף >=4800.
- `fs_hz` אמור להתאים ל-4Hz לפי runtime_config.
