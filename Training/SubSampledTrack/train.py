import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Local Import
from dataset import MelDataset

# Import PaSST
try:
    from hear21passt.base import get_basic_model
except ImportError:
    print("Please install PaSST: pip install hear21passt")
    exit()

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--mixup_alpha', type=float, default=0.4)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_DIR = Path(__file__).parent.resolve()

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

def main():
    print(f"Training on {DEVICE} in {CURRENT_DIR}")

    # Paths to the CSVs we just generated in this folder
    train_csv = CURRENT_DIR / "train.csv"
    val_csv = CURRENT_DIR / "val.csv"
    stats_json = CURRENT_DIR / "passt_stats.json"
    class_map = CURRENT_DIR / "class_map.json"

    if not train_csv.exists():
        print("ERROR: train.csv not found! Run prepare_data_preprocessed.py first.")
        return

    # 1. Load Data
    train_dataset = MelDataset(train_csv, stats_json, mode='train')
    val_dataset = MelDataset(val_csv, stats_json, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Setup Model
    with open(class_map) as f:
        num_classes = len(json.load(f))
        
    print(f"Initializing PaSST for {num_classes} classes...")
    model = get_basic_model(mode="logits")
    model.net.head = nn.Linear(768, num_classes)
    model = model.to(DEVICE)

    # 3. Setup Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs
    )

    best_acc = 0.0

    # 4. Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Mixup
            images, labels_a, labels_b, lam = mixup_data(images, labels, args.mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            
            # Approx Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (lam * predicted.eq(labels_a.data).sum().float() + (1 - lam) * predicted.eq(labels_b.data).sum().float()).item()
            
            pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CURRENT_DIR / "best_passt_model.pth")
            print(">>> New Best Model Saved!")

if __name__ == "__main__":
    main()