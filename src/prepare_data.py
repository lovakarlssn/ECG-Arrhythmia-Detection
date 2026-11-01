import os
import wfdb
import yaml
import numpy as np
import pandas as pd
import neurokit2 as nk

DS1_TRAIN_RECS = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '209', '215', '220', '223', '230']
DS1_VAL_RECS = ['201', '203', '205', '207', '208']
DS2_RECS = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'seed': 42,
        'data': {'sampling_rate': 360, 'window_before': 90, 'window_after': 90, 'lowcut': 0.5, 'highcut': 40.0}
    }

def map_aami_classes(raw_type):
    if raw_type in ['N', 'L', 'R', 'j', 'e']:
        return 'N'
    if raw_type in ['A', 'a', 'J', 'S']:
        return 'S'
    if raw_type in ['V', 'E']:
        return 'V'
    if raw_type in ['F']:
        return 'F'
    return 'X'

def download_mitbih_dataset(data_dir):
    raw_dir = os.path.join(data_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    all_recs = DS1_TRAIN_RECS + DS1_VAL_RECS + DS2_RECS
    print("Checking and downloading MIT-BIH records via wfdb...")
    for rec in all_recs:
        dat_path = os.path.join(raw_dir, f"{rec}.dat")
        if not os.path.exists(dat_path):
            try:
                wfdb.dl_database('mitdb', raw_dir, records=[rec])
            except Exception as e:
                print(f"Failed to download record {rec}: {e}")

def load_record_data(record_id, raw_dir):
    record_path = os.path.join(raw_dir, record_id)
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
    except Exception as e:
        print(f"Error loading record {record_id}: {e}")
        return None, None, None

    signal = record.p_signal[:, 0]
    if record_id == '114':
        if record.n_sig > 1:
            signal = record.p_signal[:, 1]

    peaks = annotation.sample
    symbols = annotation.symbol
    return signal, peaks, symbols

def apply_fft_bandpass(signal, fs=360, lowcut=0.5, highcut=40.0):
    signal = np.array(signal, dtype=np.float64)
    sig_fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    sig_fft[(freqs < lowcut) | (freqs > highcut)] = 0
    return np.fft.irfft(sig_fft, n=len(signal))

def normalize_beat(beat_signal):
    mean = np.mean(beat_signal)
    std = np.std(beat_signal)
    if std == 0:
        return beat_signal - mean
    return (beat_signal - mean) / std

def extract_windows(signal, peaks, symbols, window_before=90, window_after=90):
    X, y = [], []
    for peak, sym in zip(peaks, symbols):
        mapped = map_aami_classes(sym)
        if mapped == 'X':
            continue
        start = peak - window_before
        end = peak + window_after
        if start >= 0 and end < len(signal):
            window = signal[start:end]
            norm_window = normalize_beat(window)
            X.append(norm_window)
            y.append(mapped)
    return np.array(X, dtype=np.float32), np.array(y)

def match_algorithmic_peaks(expert_peaks, expert_labels, algo_peaks, tolerance=36):
    matched_algo_peaks, matched_labels = [], []
    for ep, el in zip(expert_peaks, expert_labels):
        if len(algo_peaks) == 0:
            continue
        distances = np.abs(algo_peaks - ep)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= tolerance:
            matched_algo_peaks.append(algo_peaks[closest_idx])
            matched_labels.append(el)
    return np.array(matched_algo_peaks), np.array(matched_labels)

def process_dataset():
    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    proc_dir = os.path.join(data_dir, 'processed')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    download_mitbih_dataset(data_dir)

    splits = {
        'train': DS1_TRAIN_RECS,
        'val': DS1_VAL_RECS,
        'test_expert': DS2_RECS,
        'test_algo': DS2_RECS
    }

    dataset_arrays = {}
    for split_name, recs in splits.items():
        X_list, y_list = [], []
        for rec in recs:
            signal, peaks, symbols = load_record_data(rec, raw_dir)
            if signal is None:
                continue
            cleaned = apply_fft_bandpass(signal, fs=config['data']['sampling_rate'],
                                        lowcut=config['data']['lowcut'], highcut=config['data']['highcut'])
            if split_name in ['train', 'val', 'test_expert']:
                X, y = extract_windows(cleaned, peaks, symbols,
                                       window_before=config['data']['window_before'],
                                       window_after=config['data']['window_after'])
                if len(X) > 0:
                    X_list.append(X)
                    y_list.append(y)
            elif split_name == 'test_algo':
                try:
                    _, info = nk.ecg_peaks(cleaned, sampling_rate=config['data']['sampling_rate'], method='neurokit', correct_artifacts=False)
                    algo_peaks = info['ECG_R_Peaks']
                    expert_X, expert_y = extract_windows(cleaned, peaks, symbols,
                                                         window_before=config['data']['window_before'],
                                                         window_after=config['data']['window_after'])
                    matched_peaks, matched_labels = match_algorithmic_peaks(peaks, expert_y, algo_peaks)
                    X, y = extract_windows(cleaned, matched_peaks, matched_labels,
                                           window_before=config['data']['window_before'],
                                           window_after=config['data']['window_after'])
                    if len(X) > 0:
                        X_list.append(X)
                        y_list.append(y)
                except Exception as e:
                    print(f"Algo peak detection failed for record {rec}: {e}")

        if X_list:
            final_X = np.concatenate(X_list, axis=0)
            final_y = np.concatenate(y_list, axis=0)
            dataset_arrays[split_name] = {'X': final_X, 'y': final_y}
            np.savez_compressed(os.path.join(proc_dir, f"{split_name}.npz"), X=final_X, y=final_y)
            print(f"Processed and saved {split_name}.npz with shape {final_X.shape}")

    assert set(DS1_TRAIN_RECS).isdisjoint(set(DS1_VAL_RECS)), "Data leakage between train and val records"
    assert set(DS1_TRAIN_RECS + DS1_VAL_RECS).isdisjoint(set(DS2_RECS)), "Data leakage between development and test records"
    print("Dataset preparation completed with strict split verification.")

if __name__ == '__main__':
    process_dataset()