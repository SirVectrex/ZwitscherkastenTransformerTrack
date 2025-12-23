import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import random

class MelDataset(Dataset):
    def __init__(self, csv_file, stats_file=None, target_len=1000, mode='train'):
        """
        mode: 'train' or 'val'. If 'train', applies SpecAugment.
        """
        self.data = pd.read_csv(csv_file)
        self.target_len = target_len
        self.mode = mode
        
        # PaSST Defaults (Fallback)
        self.mean = -4.268
        self.std = 4.569

        if stats_file:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.mean = stats['mean']
                self.std = stats['std']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['filepath']
        label = int(self.data.iloc[idx]['label'])
        
        # 1. LOAD uint8 DATA
        try:
            # Shape: [128, Time] or [1, 128, Time]
            mel_uint8 = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            mel_uint8 = np.zeros((128, self.target_len), dtype=np.uint8)

        # 2. DE-QUANTIZE (uint8 0-255 -> dB -80 to 0)
        # We perform this calculation in float32
        mel = mel_uint8.astype(np.float32) / 255.0  # 0.0 to 1.0
        mel = mel * 80.0 - 80.0                     # -80.0 to 0.0 dB

        # Ensure 2D [128, Time]
        if mel.ndim == 3:
            mel = mel.squeeze(0)

        # 3. LENGTH FIX (Pad/Crop to 1000)
        curr_len = mel.shape[-1]
        if curr_len < self.target_len:
            pad_width = self.target_len - curr_len
            # Reflect padding is better for audio than zeros
            mel = np.pad(mel, ((0,0), (0, pad_width)), mode='wrap') 
        elif curr_len > self.target_len:
            mel = mel[:, :self.target_len]

        # 4. PASt NORMALIZATION (Standardization)
        # (x - mean) / (std * 2) -> The *2 is a specific PaSST trick
        mel = (mel - self.mean) / (self.std * 2)

        # 5. DATA AUGMENTATION (SpecAugment) - Only for Training
        if self.mode == 'train':
            mel = self.spec_augment(mel)

        # 6. Tensor Setup [1, 128, 1000]
        mel_tensor = torch.from_numpy(mel).float()
        mel_tensor = mel_tensor.unsqueeze(0) 

        return mel_tensor, torch.tensor(label).long()

    def spec_augment(self, spec, freq_mask_param=15, time_mask_param=35):
        """
        Randomly masks vertical (time) and horizontal (freq) stripes.
        Helps prevents overfitting by forcing model to use context.
        """
        # Frequency Masking
        num_mels = spec.shape[0]
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, num_mels - f)
        spec[f0:f0+f, :] = 0  # Mask with mean (approx 0 after norm)

        # Time Masking
        num_time = spec.shape[1]
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, num_time - t)
        spec[:, t0:t0+t] = 0
        
        return spec