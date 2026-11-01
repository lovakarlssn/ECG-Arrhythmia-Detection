import os
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3}

class ECGDataset(Dataset):
    def __init__(self, npz_path, is_train=False, clamp_scale=6.0):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        self.X = data['X'].astype(np.float32)
        raw_y = data['y']
        if raw_y.dtype.kind in {'U', 'S', 'O'}:
            self.y = np.array([CLASS_MAP[str(lbl)] for lbl in raw_y], dtype=np.int64)
        else:
            self.y = raw_y.astype(np.int64)
        self.is_train = is_train
        self.clamp_scale = clamp_scale

    def __len__(self):
        return len(self.y)

    def _augment(self, beat):
        shift = np.random.randint(-5, 6)
        if shift > 0:
            beat = np.pad(beat, (shift, 0), mode='edge')[:len(beat)]
        elif shift < 0:
            beat = np.pad(beat, (0, -shift), mode='edge')[-shift:]
        scale = np.random.uniform(0.9, 1.1)
        return beat * scale

    def __getitem__(self, idx):
        beat = self.X[idx].copy()
        label = self.y[idx]
        if self.is_train:
            beat = self._augment(beat)
        if self.clamp_scale is not None and self.clamp_scale > 0:
            beat = np.clip(beat / self.clamp_scale, -1.0, 1.0)
        beat_tensor = torch.tensor(beat, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return beat_tensor, label_tensor