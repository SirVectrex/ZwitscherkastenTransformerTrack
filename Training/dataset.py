import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json

class MelDataset(Dataset):
    def __init__(self, csv_file, stats_file=None, target_len=998):
        self.data = pd.read_csv(csv_file)
        self.target_len = target_len
        
        # Standardwerte (Fallback, falls keine Datei kommt)
        self.mean = 0.0
        self.std = 1.0

        # Stats laden, falls vorhanden
        if stats_file:
            print(f"Lade Stats aus: {stats_file}")
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.mean = stats['mean']
                self.std = stats['std']
            print(f"Normalisierung aktiv: Mean={self.mean:.4f}, Std={self.std:.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['filepath']
        label = int(self.data.iloc[idx]['label'])
        
        # 1. Load data
        try:
            mel = np.load(path).astype(np.float32)
        except Exception as e:
            # Falls Datei kaputt ist, leeres Array zurückgeben (verhindert Crash)
            print(f"Fehler bei {path}: {e}")
            mel = np.zeros((128, self.target_len), dtype=np.float32)

        # 2. Dimensions Check [128, Time]
        # Falls es [1, 128, Time] ist -> [128, Time] machen für einfacheres Handling
        if mel.ndim == 3:
            mel = mel.squeeze(0)
            
        # 3. Looping (Handling short audio) - DEINE WICHTIGE LOGIK
        curr_len = mel.shape[-1]
        if curr_len < self.target_len:
            n_repeats = (self.target_len // curr_len) + 1
            # Wir nutzen numpy repeat, das ist oft stabiler hier
            mel = np.tile(mel, (1, n_repeats))
        
        # Crop to exact target length (auf 998 kürzen)
        mel = mel[:, :self.target_len]

        # 4. ECHTE NORMALISIERUNG (Das ist neu!)
        # Wir rechnen auf den Rohdaten, genau wie prepare_data es berechnet hat.
        # (Pixel - Mean) / Std
        mel = (mel - self.mean) / self.std

        # 5. Convert to Torch Tensor
        mel_tensor = torch.from_numpy(mel)
        
        # 6. Channel Dimension hinzufügen [1, 128, 998]
        # PaSST erwartet [Batch, Channel, Freq, Time]
        mel_tensor = mel_tensor.unsqueeze(0)

        return mel_tensor, torch.tensor(label).long()