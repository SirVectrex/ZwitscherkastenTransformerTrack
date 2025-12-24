import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models as models

# --- DEINE IMPORTS ---
from dataset import MelDataset

# --- KONFIGURATION ---
CSV_TRAIN = "./Training/train.csv"
CSV_VAL = "./Training/val.csv"
STATS_FILE = "/dev/shm/schoen/output/normalization_stats.json"

# WICHTIG: Hier muss die Anzahl deiner echten Klassen stehen (laut deiner Analyse 59)
NUM_CLASSES = 59         

BATCH_SIZE = 32          # MobileNet ist leicht, wir können größere Batches nehmen
LEARNING_RATE = 1e-3     # Standard Start-Rate für Transfer Learning
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "/dev/shm/schoen/checkpoints_mobilenet"
LOG_DIR = "/dev/shm/schoen/logs_mobilenet"
RUN_NAME = "mobilenet_v3_large"


def get_mobilenet_model(num_classes: int):
    """
    Lädt ein vortrainiertes MobileNetV3 und passt es an:
    1. Input: Von 3 Kanälen (RGB) auf 1 Kanal (Spektrogramm) ändern.
    2. Output: Auf deine Anzahl von Vogelarten ändern.
    """
    print(f"Lade MobileNetV3 Large (Pretrained)...")
    
    # Lade das Modell mit ImageNet-Gewichten (hilft beim schnelleren Lernen)
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # --- ANPASSUNG 1: Input Layer (von RGB zu Grayscale) ---
    # Die erste Schicht ist: Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # Wir ändern den ersten Parameter von 3 auf 1.
    original_layer = model.features[0][0]
    
    model.features[0][0] = nn.Conv2d(
        in_channels=1, 
        out_channels=original_layer.out_channels,
        kernel_size=original_layer.kernel_size,
        stride=original_layer.stride,
        padding=original_layer.padding,
        bias=False
    )
    
    # --- ANPASSUNG 2: Output Layer (Classifier Head) ---
    # Der Classifier ist ein Sequential Block. Die letzte Schicht ist Linear.
    # Wir holen uns die Eingangsgröße der letzten Schicht.
    last_layer_input = model.classifier[-1].in_features
    
    # Wir ersetzen die letzte Schicht durch eine neue mit 'num_classes' Ausgängen
    model.classifier[-1] = nn.Linear(last_layer_input, num_classes)

    return model

def save_checkpoint(path, model, optimizer, epoch, best_acc):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_acc': best_acc
    }, path)

def train_one_epoch(model, loader, criterion, optimizer, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (mels, labels) in enumerate(loop):
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)
        
        # Dimensionen prüfen: MobileNet erwartet [Batch, Channel, Height, Width]
        # Dein Dataset liefert [Batch, 128, 998] -> Wir brauchen [Batch, 1, 128, 998]
        if mels.dim() == 3:
            mels = mels.unsqueeze(1)
            
        optimizer.zero_grad()
        
        # Forward
        outputs = model(mels)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
        # Tensorboard Batch Logging
        step = epoch * len(loader) + batch_idx
        writer.add_scalar("train/batch_loss", loss.item(), step)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for mels, labels in tqdm(loader, desc="Validating"):
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)
        
        if mels.dim() == 3:
            mels = mels.unsqueeze(1)
            
        outputs = model(mels)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Tensorboard & CSV Setup
    run_id = f"{RUN_NAME}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_id))
    csv_file = os.path.join(LOG_DIR, f"{run_id}.csv")
    
    with open(csv_file, "w") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # 1. Daten laden
    print("Lade Datasets...")
    train_dataset = MelDataset(CSV_TRAIN, stats_file=STATS_FILE)
    val_dataset = MelDataset(CSV_VAL, stats_file=STATS_FILE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Modell initialisieren
    model = get_mobilenet_model(NUM_CLASSES).to(DEVICE)
    
    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Learning Rate Scheduler: Reduziert LR, wenn Loss stagniert
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"Starte Training auf {DEVICE} für {EPOCHS} Epochen...")

    for epoch in range(EPOCHS):
        # Train & Val
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, writer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Scheduler Step (basierend auf Val Accuracy)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4%}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4%}")
        print(f"LR: {current_lr:.2e}")
        print("-" * 30)
        
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        
        with open(csv_file, "a") as f:
            csv.writer(f).writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, current_lr])

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(os.path.join(SAVE_DIR, "best_mobilenet.pth"), model, optimizer, epoch, best_acc)
            print(f"--> Neues bestes Modell gespeichert! ({best_acc:.4%})")

    writer.close()
    print("Training abgeschlossen.")

if __name__ == "__main__":
    main()
