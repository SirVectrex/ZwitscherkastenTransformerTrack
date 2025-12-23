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

# Local Import
from dataset import MelDataset

# --- GLOBAL CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logic control
PHASE = 2  # 1: Head only, 2: Full Fine-Tuning
MIXUP_ALPHA = 0.4 if PHASE == 2 else 0.0

CONFIG = {
    "paths": {
        "train_csv": CURRENT_DIR / "train.csv",
        "val_csv": CURRENT_DIR / "val.csv",
        "stats_json": CURRENT_DIR / "passt_stats.json",
        "class_map": CURRENT_DIR / "class_map.json",
        "checkpoint_dir": CURRENT_DIR / "checkpoints",
        "log_dir": CURRENT_DIR / "logs",
        "load_checkpoint": CURRENT_DIR / "best_passt_model.pth" if PHASE == 2 else None
    },
    "hyperparams": {
        "batch_size": 16,
        "epochs": 20 if PHASE == 1 else 30,
        "lr": 1e-4 if PHASE == 1 else 5e-6,
        "weight_decay": 0.0001,
        "label_smoothing": 0.1,
        "max_grad_norm": 1.0,  # For stability
    }
}

# --- UTILS ---

def get_passt_model(num_classes: int):
    print(f"Initializing PaSST for {num_classes} classes...")
    model = get_basic_model(mode="logits") # Mode logits for raw output
    
    # Replace head
    in_features = model.net.head.in_features if not isinstance(model.net.head, nn.Sequential) else model.net.head[-1].in_features
    model.net.head = nn.Linear(in_features, num_classes)
    return model

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def set_patchout(model, phase):
    if phase == 2:
        model.net.s_patchout_t = 40 # Higher regularization for fine-tuning
        model.net.s_patchout_f = 6
    else:
        model.net.s_patchout_t = 20
        model.net.s_patchout_f = 4

# --- ENGINE ---

def train_one_epoch(model, loader, criterion, optimizer, scheduler, writer, epoch, global_step):
    model.train()
    running_loss, correct, total = 0.0, 0.0, 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Mixup logic
        lam = 1.0
        if PHASE == 2 and MIXUP_ALPHA > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        
        if PHASE == 2 and MIXUP_ALPHA > 0:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        
        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["hyperparams"]["max_grad_norm"])
        
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        
        # Accurate Accuracy calculation for Mixup
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if PHASE == 2 and MIXUP_ALPHA > 0:
            correct += (lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += (predicted == labels).sum().item()
            
        global_step += 1
        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)

    return running_loss / len(loader), 100. * correct / total, global_step

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return val_loss / len(loader), 100. * correct / total

# --- MAIN ---

def main():
    os.makedirs(CONFIG["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["paths"]["log_dir"], exist_ok=True)
    
    # 1. Setup Data
    with open(CONFIG["paths"]["class_map"]) as f:
        num_classes = len(json.load(f))

    train_dataset = MelDataset(CONFIG["paths"]["train_csv"], CONFIG["paths"]["stats_json"], mode='train')
    val_dataset = MelDataset(CONFIG["paths"]["val_csv"], CONFIG["paths"]["stats_json"], mode='val')

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["hyperparams"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["hyperparams"]["batch_size"], shuffle=False, num_workers=4)

    # 2. Setup Model
    model = get_passt_model(num_classes).to(DEVICE)
    set_patchout(model, PHASE)

    if PHASE == 2 and CONFIG["paths"]["load_checkpoint"]:
        print(f"Loading Phase 1 Weights from {CONFIG['paths']['load_checkpoint']}")
        model.load_state_dict(torch.load(CONFIG["paths"]["load_checkpoint"], map_location=DEVICE), strict=False)

    # 3. Freeze/Unfreeze
    if PHASE == 1:
        for name, param in model.named_parameters():
            param.requires_grad = "head" in name or "norm" in name
    else:
        for param in model.parameters():
            param.requires_grad = True

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=CONFIG["hyperparams"]["lr"], 
                            weight_decay=CONFIG["hyperparams"]["weight_decay"])
    
    # OneCycleLR is often better than Cosine for Transformer convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG["hyperparams"]["lr"] * 5, 
        steps_per_epoch=len(train_loader), 
        epochs=CONFIG["hyperparams"]["epochs"]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["hyperparams"]["label_smoothing"])
    writer = SummaryWriter(CONFIG["paths"]["log_dir"] / f"phase_{PHASE}_{int(time.time())}")

    # 5. Loop
    best_acc, global_step = 0.0, 0
    for epoch in range(CONFIG["hyperparams"]["epochs"]):
        train_loss, train_acc, global_step = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, writer, epoch, global_step)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = CONFIG["paths"]["checkpoint_dir"] / f"best_model_phase_{PHASE}.pth"
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved Best Model: {val_acc:.2f}%")

    writer.close()

if __name__ == "__main__":
    main()