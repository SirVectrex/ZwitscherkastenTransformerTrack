"""
Train MobileNet v3 for Bird Audio Classification

Uses existing preprocessed .npy files from prepare_data_from_raw_audio.py

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_name": "mobilenet_v3_small",  # or "mobilenet_v3_large"
    "num_classes": 61,
    
    "data": {
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "preprocessed_mels_dir": "preprocessed_mels",
        "class_weights_file": "class_weights_mobilenet.json",
    },
    
    "training": {
        "batch_size": 16,  # Larger batch for MobileNet (smaller model)
        "epochs": 20,
        "learning_rate": 1e-3,  # Higher LR for training from scratch
        "weight_decay": 1e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    
    "augmentation": {
        "spec_augment": True,  # Random time/frequency masking
        "mixup": True,         # Blend two samples
        "mixup_alpha": 0.2,
    },
    
    "checkpoint": {
        "save_dir": "checkpoints",
        "save_best": True,
    },
    
    "metrics": {
        "save_dir": "metrics",
    },
}

CURRENT_DIR = Path(__file__).parent.resolve()


# ============================================================================
# DATASET
# ============================================================================

class BirdAudioDataset(Dataset):
    """Load preprocessed Mel spectrograms from .npy files"""
    
    def __init__(self, csv_file, preprocessed_dir, class_weights_file=None, augment=False):
        self.df = pd.read_csv(csv_file)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.augment = augment
        
        # Load class weights
        if class_weights_file and Path(class_weights_file).exists():
            with open(class_weights_file, 'r') as f:
                weights_dict = json.load(f)
                self.class_weights = torch.tensor(
                    [weights_dict[str(i)] for i in range(len(weights_dict))],
                    dtype=torch.float32
                )
        else:
            self.class_weights = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        label = int(row['label'])
        
        # Load .npy file
        mel_spec = np.load(filepath)  # Shape: [128, T]
        mel_spec = mel_spec.astype(np.float32)
        
        # Normalize to [0, 1] range (from uint8 [0, 255])
        mel_spec = mel_spec / 255.0
        
        # Convert to 3D tensor for CNN: [1, 128, T] (add channel dimension)
        mel_spec = torch.from_numpy(mel_spec).unsqueeze(0)
        
        # Apply augmentation during training
        if self.augment:
            mel_spec = self._apply_augmentation(mel_spec)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return mel_spec, label_tensor
    
    def _apply_augmentation(self, mel_spec):
        """SpecAugment: random time/frequency masking"""
        
        # Random time masking (mask random time steps)
        if torch.rand(1).item() < 0.5:
            time_steps = mel_spec.shape[2]
            mask_width = int(time_steps * 0.1)  # Mask 10% of time
            start = torch.randint(0, max(1, time_steps - mask_width), (1,)).item()
            mel_spec[:, :, start:start+mask_width] = 0
        
        # Random frequency masking (mask random frequency bins)
        if torch.rand(1).item() < 0.5:
            freq_bins = mel_spec.shape[1]
            mask_height = int(freq_bins * 0.1)  # Mask 10% of frequency
            start = torch.randint(0, max(1, freq_bins - mask_height), (1,)).item()
            mel_spec[:, start:start+mask_height, :] = 0
        
        return mel_spec


# ============================================================================
# MODEL: MobileNet v3
# ============================================================================

class MobileNetV3Small(nn.Module):
    """Lightweight MobileNet v3 for audio classification"""
    
    def __init__(self, num_classes=61, input_channels=1):
        super().__init__()
        
        # Input: [batch, 1, 128, T]
        # Output: [batch, num_classes]
        
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            
            # Depthwise separable blocks (simplified MobileNet v3)
            self._depthwise_sep_block(16, 24, stride=1),
            self._depthwise_sep_block(24, 24, stride=2),  # Downsample time
            
            self._depthwise_sep_block(24, 40, stride=1),
            self._depthwise_sep_block(40, 40, stride=2),  # Downsample freq
            
            self._depthwise_sep_block(40, 80, stride=1),
            self._depthwise_sep_block(80, 80, stride=1),
            
            self._depthwise_sep_block(80, 112, stride=1),
            self._depthwise_sep_block(112, 112, stride=2),  # Downsample time
            
            self._depthwise_sep_block(112, 160, stride=1),
            
            # Final convolution
            nn.Conv2d(160, 960, kernel_size=1, stride=1),
            nn.BatchNorm2d(960),
            nn.Hardswish(),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )
    
    def _depthwise_sep_block(self, in_channels, out_channels, stride=1, expansion=6):
        """Inverted residual block (MobileNet v3)"""
        hidden_dim = in_channels * expansion
        
        layers = [
            # Pointwise (expand)
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(),
            
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(),
            
            # Pointwise (project)
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, class_weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for mel_specs, labels in pbar:
        mel_specs = mel_specs.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(mel_specs)
        
        # Loss with class weights
        if class_weights is not None:
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for mel_specs, labels in tqdm(val_loader, desc="Validation", leave=False):
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)
            
            logits = model(mel_specs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_predicted = torch.topk(logits, 5, dim=1)
            correct_top5 += (top5_predicted == labels.unsqueeze(1)).any(dim=1).sum().item()
            
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100 * correct / total
    avg_acc_top5 = 100 * correct_top5 / total
    
    return avg_loss, avg_acc, avg_acc_top5


def main():
    print(f"\n{'='*70}")
    print(f"🎵 MobileNet v3 Bird Audio Classification")
    print(f"{'='*70}\n")
    
    device = torch.device(CONFIG["training"]["device"])
    print(f"Using device: {device}\n")
    
    # ========== LOAD DATA ==========
    print("Loading datasets...")
    train_csv = CURRENT_DIR / CONFIG["data"]["train_csv"]
    val_csv = CURRENT_DIR / CONFIG["data"]["val_csv"]
    
    if not train_csv.exists() or not val_csv.exists():
        print(f"❌ CSV files not found!")
        print(f"   Expected: {train_csv}, {val_csv}")
        print(f"   Did you run: python prepare_data_from_raw_audio.py ?")
        return
    
    train_dataset = BirdAudioDataset(
        csv_file=str(train_csv),
        preprocessed_dir=str(CURRENT_DIR / CONFIG["data"]["preprocessed_mels_dir"]),
        class_weights_file=str(CURRENT_DIR / CONFIG["data"]["class_weights_file"]),
        augment=True
    )
    
    val_dataset = BirdAudioDataset(
        csv_file=str(val_csv),
        preprocessed_dir=str(CURRENT_DIR / CONFIG["data"]["preprocessed_mels_dir"]),
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"], 
                           shuffle=False, num_workers=2)
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}\n")
    
    # ========== BUILD MODEL ==========
    print("Building model...")
    model = MobileNetV3Small(num_classes=CONFIG["num_classes"]).to(device)
    print(f"✅ Model: MobileNet v3 Small")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,} ({trainable_params:,} trainable)\n")
    
    # ========== SETUP TRAINING ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"], 
                           weight_decay=CONFIG["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["training"]["epochs"])
    
    # Create checkpoint dir
    checkpoint_dir = CURRENT_DIR / CONFIG["checkpoint"]["save_dir"]
    checkpoint_dir.mkdir(exist_ok=True)
    
    # ========== TRAINING LOOP ==========
    best_val_acc = 0.0
    metrics_history = []
    
    print(f"Starting training for {CONFIG['training']['epochs']} epochs...\n")
    
    for epoch in range(CONFIG["training"]["epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_acc_top5 = validate(model, val_loader, criterion, device)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_acc_top5": val_acc_top5,
            "learning_rate": current_lr,
        }
        metrics_history.append(metrics)
        
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Top-5: {val_acc_top5:.2f}%")
        print(f"   LR: {current_lr:.2e}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / "best_model_mobilenet.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🌟 New best Top-1 Acc: {val_acc:.2f}% | Top-5 Acc: {val_acc_top5:.2f}% (saved)\n")
    
    # ========== SAVE METRICS ==========
    metrics_dir = CURRENT_DIR / CONFIG["metrics"]["save_dir"]
    metrics_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = metrics_dir / f"mobilenet_{timestamp}_metrics.json"
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    
    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"\nBest Val Top-1 Acc: {best_val_acc:.2f}%")
    print(f"Model saved: {checkpoint_dir / 'best_model_mobilenet.pth'}")
    print(f"Metrics saved: {metrics_file}\n")


if __name__ == "__main__":
    main()