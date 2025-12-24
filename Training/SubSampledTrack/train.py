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

# Set your current phase here
PHASE = 2  
MIXUP_ALPHA = 0.4 if PHASE == 2 else 0.0

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
        "batch_size": 4,
        "epochs": 30 if PHASE == 2 else 20,
        "lr": 1e-5 if PHASE == 2 else 1e-4,
        "weight_decay": 0.0001,
        "label_smoothing": 0.1,
        "max_grad_norm": 1.0,
    }
}

# --- MODEL UTILS ---
def get_passt_model(num_classes: int):
    print(f"Initializing PaSST for {num_classes} classes...")
    model = get_basic_model(mode="logits")
    
    # CRITICAL FIX: Bypass internal audio-to-mel conversion
    # Your dataset already provides Mel spectrograms.
    model.mel = nn.Identity()

    # Replace the classification head
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
    # 1. SETUP LOGGING AND PATHS
    CONFIG["paths"]["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["paths"]["log_dir"].mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG["paths"]["class_map"]) as f:
        num_classes = len(json.load(f))

    run_name = f"phase_{PHASE}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(CONFIG["paths"]["log_dir"] / run_name)
    csv_log_path = CONFIG["paths"]["log_dir"] / f"{run_name}_metrics.csv"
    
    with open(csv_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    # 2. MODEL INITIALIZATION
    torch.cuda.empty_cache() # Clear any residual memory
    model = get_passt_model(num_classes).to(DEVICE)
    
    if PHASE == 2:
        ckpt_path = CONFIG["paths"]["load_checkpoint"]
        if ckpt_path.exists():
            print(f"✅ Loading Phase 1 weights from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
        else:
            print(f"⚠️ No Phase 1 checkpoint found. Fine-tuning from base model.")

    # Freeze/Unfreeze Logic
    if PHASE == 1:
        print("PHASE 1: Training Head Only")
        for name, param in model.named_parameters():
            param.requires_grad = "head" in name or "norm" in name
    else:
        print("PHASE 2: Full Fine-Tuning (Memory-Optimized)")
        for param in model.parameters():
            param.requires_grad = True

    # 3. DATA LOADERS
    train_dataset = MelDataset(CONFIG["paths"]["train_csv"], CONFIG["paths"]["stats_json"], mode='train')
    val_dataset = MelDataset(CONFIG["paths"]["val_csv"], CONFIG["paths"]["stats_json"], mode='val')
    
    # We use a smaller physical batch size to avoid OOM
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["hyperparams"]["batch_size"], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["hyperparams"]["batch_size"], 
                            shuffle=False, num_workers=4)

    # 4. OPTIMIZER, SCHEDULER, CRITERION
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=CONFIG["hyperparams"]["lr"], 
                            weight_decay=CONFIG["hyperparams"]["weight_decay"])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG["hyperparams"]["lr"] * 5, 
        steps_per_epoch=len(train_loader) // (16 // CONFIG["hyperparams"]["batch_size"]), # Adjusted for accumulation
        epochs=CONFIG["hyperparams"]["epochs"]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["hyperparams"]["label_smoothing"])

    # 5. TRAINING LOOP WITH GRADIENT ACCUMULATION
    best_acc = 0.0
    global_step = 0
    # Logic: How many small batches make one "update" step
    accum_steps = 16 // CONFIG["hyperparams"]["batch_size"] 

    for epoch in range(CONFIG["hyperparams"]["epochs"]):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['hyperparams']['epochs']}")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Dimension handling for PaSST (128x998/1000)
            if images.dim() > 3:
                images = images.view(images.size(0), 128, -1)
            
            # Trim to 998 if necessary to match PaSST internal expectation exactly
            if images.shape[-1] > 998:
                images = images[:, :, :998]

            # Mixup logic
            if PHASE == 2 and MIXUP_ALPHA > 0:
                images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale loss for accumulation
            loss = loss / accum_steps
            loss.backward()

            # Weight update step
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["hyperparams"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accum_steps
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            global_step += 1
            writer.add_scalar('Loss/Train_Batch', loss.item() * accum_steps, global_step)
            pbar.set_postfix(loss=train_loss/(i+1))

        # 6. VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if images.dim() > 3:
                    images = images.view(images.size(0), 128, -1)
                if images.shape[-1] > 998:
                    images = images[:, :, :998]

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 7. PERFORMANCE TRACKING
        epoch_val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Loss/Val_Epoch', val_loss/len(val_loader), epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)

        with open(csv_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, train_loss/len(train_loader), 100*train_correct/train_total, 
                                    val_loss/len(val_loader), epoch_val_acc, current_lr])

        print(f"📊 Val Acc: {epoch_val_acc:.2f}% | Train Loss: {train_loss/len(train_loader):.4f} | LR: {current_lr:.2e}")

        # Checkpointing
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            save_path = CONFIG["paths"]["checkpoint_dir"] / f"best_model_phase_{PHASE}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🌟 Saved new best model.")

    writer.close()

if __name__ == "__main__":
    main()