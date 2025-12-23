import os
import time
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# PaSST Import
try:
    from hear21passt.base import get_basic_model
except ImportError:
    print("Please install PaSST: pip install hear21passt")
    exit()

from dataset import MelDataset

# --- GLOBAL CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PHASE = 2  

CONFIG = {
    "paths": {
        "train_csv": CURRENT_DIR / "train.csv",
        "val_csv": CURRENT_DIR / "val.csv",
        "stats_json": CURRENT_DIR / "passt_stats.json",
        "class_map": CURRENT_DIR / "class_map.json",
        "checkpoint_dir": CURRENT_DIR / "checkpoints",
        "log_dir": CURRENT_DIR / "logs",
        "load_checkpoint": CURRENT_DIR / "checkpoints" / "best_model_phase_1.pth" 
    },
    "hyperparams": {
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-5,
        "weight_decay": 0.0001,
        "label_smoothing": 0.1,
    }
}

# --- MODEL UTILS ---
def get_passt_model(num_classes: int):
    model = get_basic_model(mode="logits")
    if isinstance(model.net.head, nn.Sequential):
        in_features = model.net.head[-1].in_features
        model.net.head[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.net.head.in_features
        model.net.head = nn.Linear(in_features, num_classes)
    return model

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- TRACKING ENGINE ---
def main():
    CONFIG["paths"]["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["paths"]["log_dir"].mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG["paths"]["class_map"]) as f:
        num_classes = len(json.load(f))

    # Initialize Logging Tools
    run_name = f"phase_{PHASE}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(CONFIG["paths"]["log_dir"] / run_name)
    csv_log_path = CONFIG["paths"]["log_dir"] / f"{run_name}_metrics.csv"
    
    with open(csv_log_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    # 1. Model & Data
    model = get_passt_model(num_classes).to(DEVICE)
    
    if PHASE == 2 and CONFIG["paths"]["load_checkpoint"].exists():
        model.load_state_dict(torch.load(CONFIG["paths"]["load_checkpoint"], map_location=DEVICE), strict=False)
    
    train_dataset = MelDataset(CONFIG["paths"]["train_csv"], CONFIG["paths"]["stats_json"], mode='train')
    val_dataset = MelDataset(CONFIG["paths"]["val_csv"], CONFIG["paths"]["stats_json"], mode='val')
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["hyperparams"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["hyperparams"]["batch_size"], shuffle=False, num_workers=4)

    # 2. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["hyperparams"]["lr"], weight_decay=CONFIG["hyperparams"]["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["hyperparams"]["lr"]*5, 
                                             steps_per_epoch=len(train_loader), epochs=CONFIG["hyperparams"]["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["hyperparams"]["label_smoothing"])

    best_acc = 0.0

    # 3. Training Loop
    for epoch in range(CONFIG["hyperparams"]["epochs"]):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Mixup (Phase 2 only)
            if PHASE == 2:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item() # Approx for mixup
            
            pbar.set_postfix(loss=train_loss/len(train_loader))

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # --- Performance Summaries ---
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        # TensorBoard Logging
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # CSV Logging
        with open(csv_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc, current_lr])

        print(f"📊 Summary: Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% | LR: {current_lr:.6f}")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), CONFIG["paths"]["checkpoint_dir"] / f"best_model_phase_{PHASE}.pth")
            print(f"🌟 Best model updated!")

    writer.close()
    print(f"✅ Training complete. Logs saved to {CONFIG['paths']['log_dir']}")

if __name__ == "__main__":
    main()