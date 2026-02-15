
import os
import glob
import numpy as np
import pandas as pd
import wfdb
import json
from scipy.interpolate import interp1d

# --- Configuration ---
DATA_DIR = "/Users/tzoharlary/Documents/Projects/Hackathon/נתונים גולמיים/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0"
OUTPUT_DIR = "processed_data_v1"

# Clinical Limits
FHR_MIN, FHR_MAX = 50, 220
UC_MIN, UC_MAX = 0, 100

# Heuristics
UC_FLAT_WINDOW_SEC = 30
UC_FLAT_STD_THRESH = 0.5 

# Ensure output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def get_ph_from_header(header_path):
    try:
        with open(header_path, 'r', encoding='latin-1') as f:
            for line in f:
                if 'pH' in line:
                    clean_line = line.replace('#', '').strip()
                    parts = clean_line.split()
                    if len(parts) >= 2 and parts[0] == 'pH':
                        try:
                            return float(parts[1])
                        except ValueError:
                            pass
    except Exception:
        pass
    return None

def detect_uc_flatline_series(uc_raw, fs=4):
    """Returns boolean mask of flatline areas."""
    if len(uc_raw) == 0: return np.array([], dtype=bool)
    s = pd.Series(uc_raw)
    win_size = int(UC_FLAT_WINDOW_SEC * fs)
    rol_std = s.rolling(window=win_size, center=True, min_periods=1).std()
    rol_std = rol_std.fillna(100)
    mask_flat = ((rol_std < UC_FLAT_STD_THRESH) & (s > 1)).values
    return mask_flat

def match_lengths(fhr, uc):
    l = min(len(fhr), len(uc))
    return fhr[:l], uc[:l]

def process_pipeline():
    record_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.hea")))
    print(f"Found {len(record_paths)} records.")
    
    records_data = [] 
    max_len = 0
    
    manifest_rows = []
    
    # Global QC Stats
    qc_stats = {
        "total_records": len(record_paths),
        "records_with_ph": 0,
        "n_samples_total_capacity": 0, # Total cells in final array
        "n_samples_content": 0,        # Total valid raw length samples
        "n_samples_padding": 0,        # Total padding samples
        
        # FHR Breakdown
        "fhr_inv_total": 0,
        "fhr_inv_padding": 0,
        "fhr_inv_missing": 0,
        "fhr_inv_range": 0,
        
        # UC Breakdown
        "uc_inv_total": 0,
        "uc_inv_padding": 0,
        "uc_inv_missing": 0,
        "uc_inv_flat": 0
    }
    
    # First pass to find max length (crucial for padding calc)
    print("Scanning for max length...")
    for hea_path in record_paths:
        try:
            rec_id = os.path.basename(hea_path).replace(".hea", "")
            rec = wfdb.rdheader(os.path.join(DATA_DIR, rec_id))
            if rec.sig_len > max_len:
                max_len = rec.sig_len
        except:
            pass
            
    print(f"Max length determined: {max_len}")
    # NOTE: Max_T Strategy
    # In this training/dataset building pipeline, we dynamically determine Max_T based on the 
    # longest record in the source data. This ensures we capture all available data for training.
    # In a Real-Time Inference context, a fixed-size Ring Buffer (e.g., 20 mins) is used instead,
    # as "Max Length" has no meaning in a continuous stream.
    
    # Process records
    for hea_path in record_paths:
        rec_id = os.path.basename(hea_path).replace(".hea", "")
        ph = get_ph_from_header(hea_path)
        if ph is not None:
            qc_stats["records_with_ph"] += 1
            
        try:
            record_base = os.path.join(DATA_DIR, rec_id)
            record = wfdb.rdrecord(record_base)
            fhr_raw = record.p_signal[:, 0].astype(np.float32)
            uc_raw = record.p_signal[:, 1].astype(np.float32)
            fs = record.fs
            
            # STAGE 1 REQUIREMENT: Canonical fs must be 4Hz
            if fs != 4:
                print(f"Skipping {rec_id}: fs is {fs}, expected 4Hz (Canonical)")
                continue
            
            fhr_raw, uc_raw = match_lengths(fhr_raw, uc_raw)
            length = len(fhr_raw)
            
            # --- Disjoint QC Logic ---
            # Priority: 1. Padding (handled at save time, but tracked here implicitly for Manifest?)
            # Actually, we process content here. Padding is added later.
            # So for content: Missing > Range > Flatline.
            
            # FHR Analysis
            m_fhr_miss = (fhr_raw == 0) | np.isnan(fhr_raw)
            m_fhr_range = (fhr_raw < FHR_MIN) | (fhr_raw > FHR_MAX)
            
            # Disjoint counts
            cnt_fhr_miss = m_fhr_miss.sum()
            # Range is valid only if NOT missing
            cnt_fhr_range = (m_fhr_range & ~m_fhr_miss).sum()
            
            mask_fhr_content = m_fhr_miss | m_fhr_range
            
            # UC Analysis
            m_uc_miss = np.isnan(uc_raw) # UC 0 is valid usually? Need clarify. Assuming 0 is valid unless Flatline logic catches it.
            # User prompted "E2... uc==0 valid?". Let's assume Valid, but Flatline logic handles "series of 0".
            # Actually Flatline logic: (std < T) & (s > 1). So 0 is NOT flatline by this def.
            # Let's keep it simple for V1 certification: NaN is missing.
            
            m_uc_flat = detect_uc_flatline_series(uc_raw, fs)
            
            cnt_uc_miss = m_uc_miss.sum()
            cnt_uc_flat = (m_uc_flat & ~m_uc_miss).sum()
            
            mask_uc_content = m_uc_miss | m_uc_flat
            
            # --- Imputation (Content Only) ---
            # FHR
            x_fhr = fhr_raw.copy()
            x_fhr[mask_fhr_content] = np.nan
            valid_idx = np.where(~np.isnan(x_fhr))[0]
            if len(valid_idx) > 1:
                f = interp1d(valid_idx, x_fhr[valid_idx], kind='linear', fill_value="extrapolate")
                x_fhr_filled = f(np.arange(length))
            else:
                x_fhr_filled = np.zeros(length)
                
            # UC
            x_uc = uc_raw.copy()
            x_uc[mask_uc_content] = np.nan
            valid_idx_uc = np.where(~np.isnan(x_uc))[0]
            if len(valid_idx_uc) > 1:
                f = interp1d(valid_idx_uc, x_uc[valid_idx_uc], kind='linear', fill_value="extrapolate")
                x_uc_filled = f(np.arange(length))
            else:
                x_uc_filled = np.zeros(length)
                
            # --- Norm ---
            x_fhr_filled = np.clip(x_fhr_filled, FHR_MIN, FHR_MAX)
            x_uc_filled = np.clip(x_uc_filled, UC_MIN, UC_MAX)
            
            x_fhr_norm = (x_fhr_filled - FHR_MIN) / (FHR_MAX - FHR_MIN)
            x_uc_norm = (x_uc_filled - UC_MIN) / (UC_MAX - UC_MIN)
            
            x_final = np.stack([x_fhr_norm, x_uc_norm], axis=1)
            
            # --- Padding Stats Calculation ---
            n_pad = max_len - length
            
            # Update Global Counters
            qc_stats["n_samples_content"] += length
            qc_stats["n_samples_padding"] += n_pad
            
            # FHR
            qc_stats["fhr_inv_missing"] += int(cnt_fhr_miss)
            qc_stats["fhr_inv_range"] += int(cnt_fhr_range)
            qc_stats["fhr_inv_padding"] += int(n_pad)
            qc_stats["fhr_inv_total"] += int(cnt_fhr_miss + cnt_fhr_range + n_pad)
            
            # UC
            qc_stats["uc_inv_missing"] += int(cnt_uc_miss)
            qc_stats["uc_inv_flat"] += int(cnt_uc_flat)
            qc_stats["uc_inv_padding"] += int(n_pad)
            qc_stats["uc_inv_total"] += int(cnt_uc_miss + cnt_uc_flat + n_pad)
            
            records_data.append({
                "id": rec_id,
                "x": x_final,
                "mask_fhr": mask_fhr_content,
                "mask_uc": mask_uc_content,
                "length": length
            })
            
            manifest_rows.append({
                "record_id": rec_id,
                "samples": length,
                "duration_sec": length / fs,
                "ph": ph,
                "padding_samples": n_pad,
                "fhr_inv_content_pct": mask_fhr_content.mean(),
                "uc_inv_content_pct": mask_uc_content.mean()
            })
            
        except Exception as e:
            print(f"Error {rec_id}: {e}")

    # --- Save Artifacts ---
    N = len(records_data)
    qc_stats["n_samples_total_capacity"] = N * max_len
    
    X_all = np.zeros((N, max_len, 2), dtype=np.float32)
    # Init masks as True (Invalid) to cover padding automatically
    mask_fhr_all = np.ones((N, max_len), dtype=bool)
    mask_uc_all = np.ones((N, max_len), dtype=bool)
    
    id_map = {}
    
    for i, item in enumerate(records_data):
        l = item["length"]
        
        # Copy content
        X_all[i, :l, :] = item["x"]
        
        # Copy content masks (False=Valid, True=Invalid)
        # Note: Padding remainder remains 1 (True/Invalid) from init
        mask_fhr_all[i, :l] = item["mask_fhr"]
        mask_uc_all[i, :l] = item["mask_uc"]
        
        id_map[item["id"]] = i
        
    print("Saving...")
    np.save(os.path.join(OUTPUT_DIR, "X_norm_4hz.npy"), X_all)
    np.save(os.path.join(OUTPUT_DIR, "mask_fhr.npy"), mask_fhr_all)
    np.save(os.path.join(OUTPUT_DIR, "mask_uc.npy"), mask_uc_all)
    
    df = pd.DataFrame(manifest_rows)
    df["array_idx"] = df["record_id"].map(id_map)
    df.to_csv(os.path.join(OUTPUT_DIR, "manifest.csv"), index=False)
    
    with open(os.path.join(OUTPUT_DIR, "qc_report.json"), "w") as f:
        json.dump(qc_stats, f, indent=2, cls=NumpyEncoder)
        
    print("Done.")

if __name__ == "__main__":
    process_pipeline()
