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

from hear21passt.base import get_basic_model
from tqdm import tqdm

# --- IMPORTS FROM YOUR FILES ---
from dataset import MelDataset

# --- CONFIGURATION ---
CSV_TRAIN = "./Training/train.csv"  # columns: filepath, label
CSV_VAL = "./Training/val.csv"
NUM_CLASSES = 59         # Change this to your number of bird species
BATCH_SIZE = 16          # PaSST is heavy; reduce this if you get CUDA OOM
LEARNING_RATE = 1e-3     # Initial Value (will be overwritten if Phase 2)
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "/dev/shm/schoen/checkpoints"
LOG_DIR = "/dev/shm/schoen/logs"
RUN_NAME = "passt_finetune"

SAVE_EVERY_EPOCH = True        # if disk is an issue set False
EARLY_STOP_PATIENCE = 5        # epochs without val_acc improvement before stopping

PHASE = 2
LOAD_CHECKPOINT = None  # Default init

# --- CONFIG PHASE 2 ---
if PHASE == 2:
    #LOAD_CHECKPOINT = "/dev/shm/schoen/checkpoints/Bestmodell_phase1.pth" 
    LEARNING_RATE = 1e-5    
    EPOCHS = 20             

def get_passt_model(num_classes: int):
    print("Loading PaSST model...")
    model = get_basic_model(mode="all", arch="passt_s_swa_p16_128_ap476")

    # 1. Bypass the Audio Frontend (your dataset already provides mels)
    model.mel = nn.Identity()

    # 2. Replace the Classification Head
    if isinstance(model.net.head, nn.Sequential):
        in_features = model.net.head[-1].in_features
        model.net.head[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.net.head.in_features
        model.net.head = nn.Linear(in_features, num_classes)

    return model


def get_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, best_acc: float, extra: dict | None = None):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # Nutzt automatisch das gleiche Device wie die Input-Daten (CPU oder CUDA)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    writer: SummaryWriter | None = None,
                    global_step_start: int = 0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    global_step = global_step_start

    loop = tqdm(loader, desc="Training")
    for mels, labels in loop:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Expecting [B, 1, 128, 998] or [B, 128, 998]
        if mels.dim() == 4:
            mels = mels.squeeze(1)  # [B, 128, 998]
        elif mels.dim() != 3:
            raise ValueError(f"Unexpected mel shape: {tuple(mels.shape)}")
        
        if PHASE == 1:
            logits = model(mels)
            loss = criterion(logits, labels)
            # Für Accuracy Berechnung unten vorbereiten
            targets = labels

        if PHASE == 2:
            # ---------------- MIXUP START ----------------
            alpha_val = 0.4
            mels_mixed, labels_a, labels_b, lam = mixup_data(mels, labels, alpha=alpha_val)
        
            logits = model(mels_mixed)
        
            # Loss berechnen (Mischung aus Loss A und Loss B)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            # ---------------- MIXUP ENDE -----------------
            
            # Für Accuracy: Wir brauchen wieder EINE Wahrheit (die stärkere Klasse)
            # Da labels hier noch die originalen Indizes sind, ist 'labels' eigentlich okay,
            # aber targets für den Vergleich unten definieren:
            targets = labels 

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy Berechnung
        _, predicted = torch.max(logits, 1)
        
        # Sicherheits-Check falls Labels doch mal One-Hot wären (bei Mixup V2)
        if len(targets.size()) > 1: 
             _, final_targets = torch.max(targets, 1)
        else:
             final_targets = targets
        
        total += final_targets.size(0)
        correct += (predicted == final_targets).sum().item()

        if writer is not None:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)

        global_step += 1
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc, global_step


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for mels, labels in loader:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if mels.dim() == 4:
            mels = mels.squeeze(1)
        elif mels.dim() != 3:
            raise ValueError(f"Unexpected mel shape: {tuple(mels.shape)}")

        logits = model(mels)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


def set_patchout_difficulty(model: nn.Module, difficulty: str = "standard"):
    """
    Adjusts the structured Patchout (dropping time/freq strips).
    """
    if difficulty == "hard":
        model.net.s_patchout_t = 40
        model.net.s_patchout_f = 6
        print("Patchout set to HARD (High regularization)")
    elif difficulty == "light":
        model.net.s_patchout_t = 10
        model.net.s_patchout_f = 2
        print("Patchout set to LIGHT (Low regularization)")
    else:
        model.net.s_patchout_t = 20
        model.net.s_patchout_f = 4
        print("Patchout set to STANDARD")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    run_id = f"{RUN_NAME}_{time.strftime('%Y%m%d-%H%M%S')}"
    tb_dir = os.path.join(LOG_DIR, run_id)
    writer = SummaryWriter(log_dir=tb_dir)

    csv_path = os.path.join(LOG_DIR, f"{run_id}.csv")
    best_meta_path = os.path.join(LOG_DIR, f"{run_id}_best.json")

    # 1. Data Setup
    print("Setting up data...")
    STATS_FILE = "/dev/shm/schoen/output/normalization_stats.json"
    
    train_dataset = MelDataset(CSV_TRAIN, stats_file=STATS_FILE)
    val_dataset = MelDataset(CSV_VAL, stats_file=STATS_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Model Setup
    model = get_passt_model(NUM_CLASSES).to(DEVICE)

    # 2.5 Optional: Adjust Patchout Difficulty
    if PHASE == 2:
        set_patchout_difficulty(model, difficulty="hard")
    else:
        set_patchout_difficulty(model, difficulty="standard")

    if LOAD_CHECKPOINT:
        print(f"--- Lade Phase 1: {LOAD_CHECKPOINT} ---")
        # map_location sorgt dafür, dass es auch lädt, wenn GPUs unterschiedlich sind
        checkpoint = torch.load(LOAD_CHECKPOINT, map_location=DEVICE)
        
        # Schlüssel anpassen, falls nötig (z.B. wenn Checkpoint 'module.' prefix hat)
        state_dict = checkpoint["model_state"]
        # model.load_state_dict(state_dict) # Strict=True ist standard, False ist sicherer bei kleinen Abweichungen
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warnung beim Laden (versuche strict=False): {e}")
            model.load_state_dict(state_dict, strict=False)

    if PHASE != 2:
        # --- FREEZING (Nur Phase 1) ---
        print("Friere das Basis-Modell ein. Trainiere nur den Kopf...")
        for name, param in model.named_parameters():
            if "head" in name:
                param.requires_grad = True
            elif "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainierbare Parameter: {trainable_params} von {total_params} ({(trainable_params/total_params):.1%})")
    else:
        # --- UNFREEZE (Phase 2) ---
        print("Phase 2: Unfreeze All Parameters for Fine-Tuning")
        for param in model.parameters():
            param.requires_grad = True

    # 3. Optimizer & Loss
    # Wichtig: optimizer darf nur Parameter bekommen, die requires_grad=True haben
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # CSV header
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_sec"])

    print(f"Starting training on {DEVICE}...")
    best_acc = 0.0
    best_epoch = -1
    epochs_since_improve = 0
    global_step = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, writer=writer, global_step_start=global_step
        )
        val_loss, val_acc = validate(model, val_loader, criterion)

        epoch_time = time.time() - t0
        
        scheduler.step()
        lr = get_lr(optimizer)

        # Console
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")

        # TensorBoard
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/epoch_acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        # CSV logging
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, lr, epoch_time])

        # Save checkpoint
        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                os.path.join(SAVE_DIR, f"epoch_{epoch+1:03d}.pth"),
                model, optimizer, epoch + 1, best_acc
            )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_since_improve = 0

            save_checkpoint(
                os.path.join(SAVE_DIR, "best_model.pth"),
                model, optimizer, epoch + 1, best_acc,
                extra={"val_acc": val_acc}
            )
            print(f"Saved Best Model (epoch {best_epoch})!")
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping: no val_acc improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

        print("-" * 30)

    writer.close()
    print("Done.")

if __name__ == "__main__":
    main()
