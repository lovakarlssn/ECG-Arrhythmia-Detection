import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from dataset import ECGDataset, CLASS_MAP
from model import MAX78002_1DCNN, audit_model_memory

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'seed': 42,
        'training': {'batch_size': 128, 'epochs_fp32': 35, 'epochs_qat': 15, 'lr': 0.001, 'weight_decay': 0.0001, 'focal_gamma': 2.0, 'focal_power': 0.5, 'patience': 8}
    }

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = torch.sum(probs * targets_one_hot, dim=1)
        log_pt = torch.sum(log_probs * targets_one_hot, dim=1)
        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        loss = -focal_weight * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

def compute_class_weights(y_train, power=0.5):
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    num_classes = len(CLASS_MAP)
    weights = np.ones(num_classes, dtype=np.float32)
    for c, count in zip(classes, counts):
        weights[c] = (total / (num_classes * max(count, 1))) ** power
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, macro_f1

def main():
    config = load_config()
    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    train_ds = ECGDataset(os.path.join(proc_dir, 'train.npz'), is_train=True)
    val_ds = ECGDataset(os.path.join(proc_dir, 'val.npz'), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)

    alpha_weights = compute_class_weights(train_ds.y, power=config['training']['focal_power']).to(device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=config['training']['focal_gamma'])

    model = MAX78002_1DCNN(num_classes=4, input_channels=1,
                           base_filters=config.get('model', {}).get('base_filters', 16),
                           dropout=config.get('model', {}).get('dropout', 0.1)).to(device)
    audit_model_memory(model)
    model.set_quantization(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = -1.0
    patience_counter = 0
    fp32_ckpt = os.path.join(results_dir, 'best_model.pt')
    int8_ckpt = os.path.join(results_dir, 'best_model_int8.pt')

    for epoch in range(1, config['training']['epochs_fp32'] + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        _, val_acc, val_f1 = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), fp32_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Loading best FP32 model with Val Macro F1: {best_val_f1:.4f}")
    model.load_state_dict(torch.load(fp32_ckpt, map_location=device))

    print("Starting Quantization-Aware Training (QAT)...")
    model.set_quantization(True)
    qat_optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'] * 0.1, weight_decay=config['training']['weight_decay'])

    best_qat_f1 = -1.0
    for epoch in range(1, config['training']['epochs_qat'] + 1):
        model.train()
        qat_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            qat_optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            qat_optimizer.step()
            qat_loss += loss.item() * x.size(0)
        qat_loss /= len(train_loader.dataset)
        _, val_acc, val_f1 = evaluate_loader(model, val_loader, criterion, device)
        print(f"QAT Epoch {epoch:02d} | Loss: {qat_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 >= best_qat_f1:
            best_qat_f1 = val_f1
            torch.save(model.state_dict(), int8_ckpt)

    print(f"Training complete. Saved INT8 QAT model to {int8_ckpt}")

if __name__ == '__main__':
    main()