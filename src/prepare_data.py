import os
import pandas as pd
import numpy as np
import neurokit2 as nk

DS1_TRAIN_RECS = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '209', '215', '220', '223', '230']
DS1_VAL_RECS = ['201', '203', '205', '207', '208']
DS2_RECS = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

def map_aami_classes(raw_type):
    if raw_type in ['N', 'L', 'R', 'e', 'j']: return 'N'
    if raw_type in ['A', 'a', 'J', 'S']: return 'S'
    if raw_type in ['V', 'E']: return 'V'
    if raw_type in ['F']: return 'F'
    return 'X'

def load_record(record_id, data_dir):
    csv_path = os.path.join(data_dir, f"{record_id}.csv")
    txt_path = os.path.join(data_dir, f"{record_id}annotations.txt")
    
    if not os.path.exists(csv_path) or not os.path.exists(txt_path):
        print(f"  [!] Missing files for record {record_id} in {data_dir}. Skipping.")
        return None, None, None
        
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().strip("'") for col in df.columns]
    
    if 'MLII' in df.columns:
        signal = df['MLII'].values
    elif 'V5' in df.columns:
        signal = df['V5'].values
    else:
        signal = df.iloc[:, 1].values
        
    peaks = []
    labels = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            mapped_label = map_aami_classes(parts[2])
            if mapped_label != 'X':
                peaks.append(int(parts[1]))
                labels.append(mapped_label)
                
    return signal, np.array(peaks), np.array(labels)

def apply_fft_bandpass(signal, fs=360, lowcut=0.5, highcut=40.0):
    signal = np.array(signal, dtype=np.float64)
    sig_fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    sig_fft[(freqs < lowcut) | (freqs > highcut)] = 0
    filtered_sig = np.fft.irfft(sig_fft, n=len(signal))
    return filtered_sig

def normalize_beat(beat_signal):
    mean = np.mean(beat_signal)
    std = np.std(beat_signal)
    if std == 0:
        return beat_signal - mean
    return (beat_signal - mean) / std

def extract_windows(signal, peaks, labels, window_before=90, window_after=90):
    X, y = [], []
    for idx, peak in enumerate(peaks):
        start = peak - window_before
        end = peak + window_after
        if start >= 0 and end < len(signal):
            window = signal[start:end]
            normalized_window = normalize_beat(window)
            X.append(normalized_window)
            y.append(labels[idx])
    return np.array(X), np.array(y)

def match_algorithmic_peaks(expert_peaks, expert_labels, algo_peaks, tolerance=36):
    matched_algo_peaks = []
    matched_labels = []
    
    for idx, ep in enumerate(expert_peaks):
        if len(algo_peaks) == 0:
            continue
        distances = np.abs(algo_peaks - ep)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= tolerance:
            matched_algo_peaks.append(algo_peaks[closest_idx])
            matched_labels.append(expert_labels[idx])
            
    return np.array(matched_algo_peaks), np.array(matched_labels)

def process_dataset():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data") if os.path.basename(os.getcwd()) == "src" else "data"
    out_dir = "processed"
    results_dir = "results"
    
    print(f"Looking for data in: {os.path.abspath(data_dir)}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    datasets = {
        "train": {"recs": DS1_TRAIN_RECS, "X": [], "y": []},
        "val": {"recs": DS1_VAL_RECS, "X": [], "y": []},
        "test_expert": {"recs": DS2_RECS, "X": [], "y": []},
        "test_algo": {"recs": DS2_RECS, "X": [], "y": []}
    }
    
    stats_lines = ["Final Dataset Statistics\n", "-"*30 + "\n"]
    
    for split_name in ["train", "val", "test"]:
        recs = datasets[split_name]["recs"] if split_name != "test" else DS2_RECS
        print(f"\n--- Processing {split_name.upper()} split ({len(recs)} records) ---")
        
        for rec in recs:
            print(f"  -> Loading {rec}...", end="", flush=True)
            signal, expert_peaks, expert_labels = load_record(rec, data_dir)
            
            if signal is None:
                print(" Failed (File not found).")
                continue
                
            print(" Bandpass (FFT)...", end="", flush=True)
            cleaned_signal = apply_fft_bandpass(signal, fs=360)
            
            if split_name in ["train", "val"]:
                print(" Extracting...", end="", flush=True)
                X, y = extract_windows(cleaned_signal, expert_peaks, expert_labels)
                datasets[split_name]["X"].append(X)
                datasets[split_name]["y"].append(y)
                print(f" Extracted {len(y)} beats.")
                
            elif split_name == "test":
                print(" Extract Expert...", end="", flush=True)
                X_exp, y_exp = extract_windows(cleaned_signal, expert_peaks, expert_labels)
                datasets["test_expert"]["X"].append(X_exp)
                datasets["test_expert"]["y"].append(y_exp)
                
                print(" Detect Algo...", end="", flush=True)
                try:
                    _, info = nk.ecg_peaks(cleaned_signal, sampling_rate=360, method="neurokit", correct_artifacts=False)
                    algo_peaks = info["ECG_R_Peaks"]
                    matched_algo_peaks, matched_labels = match_algorithmic_peaks(expert_peaks, expert_labels, algo_peaks)
                    X_alg, y_alg = extract_windows(cleaned_signal, matched_algo_peaks, matched_labels)
                    datasets["test_algo"]["X"].append(X_alg)
                    datasets["test_algo"]["y"].append(y_alg)
                    print(f" Done ({len(y_exp)} expert, {len(y_alg)} algo).")
                except Exception as e:
                    print(f" [Algo Failed: {e}].")
    
    print("\n--- Saving Arrays to Disk ---")
    for key in datasets.keys():
        if len(datasets[key]["X"]) > 0:
            final_X = np.concatenate(datasets[key]["X"], axis=0)
            final_y = np.concatenate(datasets[key]["y"], axis=0)
            file_path = os.path.join(out_dir, f"{key}.npz")
            np.savez_compressed(file_path, X=final_X, y=final_y)
            print(f"Saved {file_path} (Shape: {final_X.shape})")
            
            unique, counts = np.unique(final_y, return_counts=True)
            dist = dict(zip(unique, counts))
            stats_lines.append(f"{key.upper()} Split - Total Beats: {len(final_y)}")
            for cls in ['N', 'S', 'V', 'F']:
                count = dist.get(cls, 0)
                stats_lines.append(f"  {cls}: {count} ({(count/len(final_y))*100:.2f}%)")
            stats_lines.append("\n")

    stats_file = os.path.join(results_dir, "dataset_statistics.txt")
    with open(stats_file, "w") as f:
        f.writelines(stats_lines)
        
    print(f"\nSUCCESS! All processing complete. Statistics saved to {stats_file}.")

if __name__ == "__main__":
    process_dataset()