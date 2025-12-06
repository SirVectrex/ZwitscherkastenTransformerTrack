import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class MelDataset(Dataset):
    def __init__(self, csv_file, target_len=998):
        self.data = pd.read_csv(csv_file)
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['filepath']
        label = self.data.iloc[idx]['label']
        
        # 1. Load uint8 data [Shape: 128, Time]
        mel = np.load(path)
        
        # 2. CONVERT UINT8 TO FLOAT
        # We assume uint8 maps 0->255. We scale to 0.0->1.0
        mel = mel.astype(np.float32) / 255.0
        
        # --- dataset.py - Inside MelDataset's __getitem__ method ---

        # ... (After loading and converting to float) ...
        
        # 3. Convert to Torch
        mel = torch.from_numpy(mel)
        
        # 4. Remove any existing extra dims, just in case
        # This addresses potential issues from how the NPY files were saved.
        if mel.ndim == 3 and mel.shape[0] == 1 and mel.shape[1] == 128:
            # If shape is already [1, 128, T], we are fine.
            pass
        
        # 5. Ensure Channel Dimension is CORRECTLY [1, 128, Time]
        # If the shape is [128, Time] (most common way to save a 2D spectrogram),
        # we unsqueeze the channel dimension (C=1) at the front.
        if mel.ndim == 2:
            mel = mel.unsqueeze(0) # Shape becomes [1, 128, Time]
        
        # CRITICAL: If your NPY was saved as a 3D array [1, 128, Time] 
        # but the DataLoader is still adding a dim, let's fix the squeeze now:
        # We will remove any dimension where the size is 1, except for the first one 
        # which is the true channel dimension.

        # 6. Final Clean-up (Preventing the error)
        # The redundant '1' dimension is being inserted somewhere. 
        # We use .squeeze() to remove all dimensions of size 1.
        mel = mel.squeeze()
        
        # After squeezing, ensure shape is exactly [1, 128, Time]
        if mel.ndim == 2:
            # If it was fully squeezed down to 2D, restore the channel dimension
            mel = mel.unsqueeze(0)
        
        # ... (Continue with the LOOPING and Normalization logic) ...

        # ... (Rest of the function remains the same) ...

        # 5. LOOPING (Handling short audio)
        curr_len = mel.shape[-1]
        if curr_len < self.target_len:
            n_repeats = (self.target_len // curr_len) + 1
            mel = mel.repeat(1, 1, n_repeats)
        
        # Crop to exact target length
        mel = mel[:, :, :self.target_len]

        # 6. PaSST Normalization
        # PaSST was trained on AudioSet with specific normalization.
        # Since we just converted 0-255 to 0-1, we should shift it to be roughly -1 to 1.
        # (This is a standard approximation for PaSST inputs)
        mel = (mel * 2.0) - 1.0


        return mel, torch.tensor(label).long()