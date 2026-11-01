import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import ECGDataset
from model import MAX78002_1DCNN
from prepare_data import DS2_RECS, load_record_data, apply_fft_bandpass, normalize_beat, map_aami_classes

def download_nstdb(data_dir):
    nstdb_dir = os.path.join(data_dir, 'raw', 'nstdb')
    os.makedirs(nstdb_dir, exist_ok=True)
    for rec in ['bw', 'ma', 'em']:
        dat_path = os.path.join(nstdb_dir, f"{rec}.dat")
        if not os.path.exists(dat_path):
            try:
                wfdb.dl_database('nstdb', nstdb_dir, records=[rec])
            except Exception as e:
                print(f"Failed to download NSTDB record {rec}: {e}")

def add_noise(signal, noise_source, snr_db):
    sig_power = np.mean(signal ** 2)
    if sig_power == 0:
        return signal
    if len(noise_source) < len(signal):
        noise_source = np.tile(noise_source, int(np.ceil(len(signal) / len(noise_source))))
    noise_seg = noise_source[:len(signal)]
    noise_power = np.mean(noise_seg ** 2)
    if noise_power == 0:
        return signal
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return signal + scale * noise_seg

def run_noise_test():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    download_nstdb(data_dir)

    nstdb_dir = os.path.join(data_dir, 'raw', 'nstdb')
    try:
        noise_rec = wfdb.rdrecord(os.path.join(nstdb_dir, 'bw'))
        noise_source = noise_rec.p_signal[:, 0]
    except Exception:
        noise_source = np.random.randn(360000) * 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAX78002_1DCNN(num_classes=4, input_channels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_int8.pt'), map_location=device))
    model.set_quantization(True)
    model.eval()

    snr_levels = [None, 12, 6, 0]
    condition_names = ['Clean', '12 dB', '6 dB', '0 dB']
    results = []

    raw_dir = os.path.join(data_dir, 'raw')
    for snr, cond in zip(snr_levels, condition_names):
        X_list, y_list = [], []
        for rec in DS2_RECS[:5]:
            signal, peaks, symbols = load_record_data(rec, raw_dir)
            if signal is None:
                continue
            if snr is not None:
                signal = add_noise(signal, noise_source, snr)
            cleaned = apply_fft_bandpass(signal, fs=360, lowcut=0.5, highcut=40.0)
            for peak, sym in zip(peaks, symbols):
                mapped = map_aami_classes(sym)
                if mapped == 'X':
                    continue
                start = peak - 90
                end = peak + 90
                if start >= 0 and end < len(cleaned):
                    window = cleaned[start:end]
                    norm_window = normalize_beat(window)
                    X_list.append(norm_window)
                    y_list.append({'N': 0, 'S': 1, 'V': 2, 'F': 3}[mapped])
        if len(X_list) > 0:
            X_arr = np.array(X_list, dtype=np.float32)
            y_arr = np.array(y_list, dtype=np.int64)
            dataset = torch.utils.data.TensorDataset(torch.tensor(X_arr).unsqueeze(1), torch.tensor(y_arr))
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            preds, targets = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    p = torch.argmax(model(x), dim=1)
                    preds.append(p.cpu().numpy())
                    targets.append(y.numpy())
            y_true = np.concatenate(targets)
            y_pred = np.concatenate(preds)
            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            results.append({'Condition': cond, 'Accuracy': acc, 'Macro_F1': macro_f1})

    res_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(res_df['Condition'], res_df['Macro_F1'], marker='o', linewidth=2, color='darkred')
    plt.title('Noise Robustness Degradation Curve (INT8 Model)')
    plt.xlabel('SNR Condition')
    plt.ylabel('Macro F1 Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'noise_results.png'), dpi=300)
    plt.close()
    print("Noise test completed and noise_results.png saved.")

if __name__ == '__main__':
    run_noise_test()