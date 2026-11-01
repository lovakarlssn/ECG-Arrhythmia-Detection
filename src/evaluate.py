import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from dataset import ECGDataset, CLASS_MAP
from model import MAX78002_1DCNN, BaselineClassifier

def evaluate_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    results_dir = os.path.join(base_dir, 'results')
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = ECGDataset(os.path.join(proc_dir, 'test_expert.npz'), is_train=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    train_ds = ECGDataset(os.path.join(proc_dir, 'train.npz'), is_train=False)

    baseline = BaselineClassifier()
    baseline.fit(train_ds.X, train_ds.y)
    baseline_preds = baseline.predict(test_ds.X)
    baseline_acc = accuracy_score(test_ds.y, baseline_preds)
    baseline_f1 = f1_score(test_ds.y, baseline_preds, average='macro', zero_division=0)

    fp32_model = MAX78002_1DCNN(num_classes=4, input_channels=1).to(device)
    fp32_model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pt'), map_location=device))
    fp32_model.set_quantization(False)
    fp32_model.eval()

    int8_model = MAX78002_1DCNN(num_classes=4, input_channels=1).to(device)
    int8_model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_int8.pt'), map_location=device))
    int8_model.set_quantization(True)
    int8_model.eval()

    def get_nn_preds(model):
        preds, targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                p = torch.argmax(logits, dim=1)
                preds.append(p.cpu().numpy())
                targets.append(y.numpy())
        return np.concatenate(targets), np.concatenate(preds)

    y_true, fp32_preds = get_nn_preds(fp32_model)
    _, int8_preds = get_nn_preds(int8_model)

    fp32_acc = accuracy_score(y_true, fp32_preds)
    fp32_f1 = f1_score(y_true, fp32_preds, average='macro', zero_division=0)

    int8_acc = accuracy_score(y_true, int8_preds)
    int8_f1 = f1_score(y_true, int8_preds, average='macro', zero_division=0)

    metrics_data = [
        {"Model": "Baseline (Logistic Regression)", "Accuracy": baseline_acc, "Macro_F1": baseline_f1, "Footprint_KB": f"{baseline.get_params_count()*4/1024:.2f} KB"},
        {"Model": "FP32 Edge Model (1D CNN)", "Accuracy": fp32_acc, "Macro_F1": fp32_f1, "Footprint_KB": "88.98 KB"},
        {"Model": "Quantized Model (INT8 CNN)", "Accuracy": int8_acc, "Macro_F1": int8_f1, "Footprint_KB": "22.24 KB"}
    ]
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    print("Metrics summary:")
    print(metrics_df.to_string(index=False))

    cm = confusion_matrix(y_true, int8_preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['N', 'S', 'V', 'F'], yticklabels=['N', 'S', 'V', 'F'])
    plt.title('Confusion Matrix - INT8 Quantized Edge Model')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6), dpi=300)
    class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']
    for i, name in enumerate(class_names):
        subset = test_ds.X[test_ds.y == i]
        if len(subset) > 0:
            mean_beat = np.mean(subset, axis=0).squeeze()
            plt.plot(mean_beat, label=name, linewidth=2)
    plt.title('Mean Normalized Beat Morphologies per Class')
    plt.xlabel('Samples (0.5s at 360Hz)')
    plt.ylabel('Z-Score Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'mean_beat_morphologies.png'), dpi=300)
    plt.close()

    print("Evaluation completed and figures saved.")

if __name__ == '__main__':
    evaluate_model()