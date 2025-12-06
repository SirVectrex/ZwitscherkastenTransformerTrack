import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from hear21passt.base import get_basic_model
from tqdm import tqdm
import os

# --- IMPORTS FROM YOUR FILES ---
from dataset import MelDataset

# --- CONFIGURATION ---
CSV_TRAIN = "train.csv"  # columns: filepath, label
CSV_VAL = "val.csv"
NUM_CLASSES = 50         # Change this to your number of bird species
BATCH_SIZE = 16          # PaSST is heavy; reduce this if you get CUDA OOM
LEARNING_RATE = 1e-5     # Low LR is best for finetuning
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"

def get_passt_model(num_classes):
    # 1. Load Pretrained PaSST
    # mode="all" enables Patchout (randomly drops patches to speed up training)
    # arch="passt_s_swa_p16_128_ap476" is the high-performance AudioSet checkpoint
    print("Loading PaSST model...")
    model = get_basic_model(mode="all", arch="passt_s_swa_p16_128_ap476")

    # 2. CRITICAL: Bypass the Audio Frontend
    # The original model expects raw audio and converts it to Mel.
    # We replace that conversion layer with Identity since we feed Mels directly.
    model.mel = nn.Identity()

    # 3. Replace the Classification Head
    # PaSST's classifier is stored in model.net.head
    in_features = model.net.head.in_features
    model.net.head = nn.Linear(in_features, num_classes)
    
    return model

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training")
    for mels, labels in loop:
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        
        # Forward pass
        # Input shape: [Batch, 1, 128, 998]
        logits = model(mels) 
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Accuracy tracking
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mels, labels in loader:
            mels, labels = mels.to(DEVICE), labels.to(DEVICE)
            
            # Note: PaSST usually disables Patchout during eval automatically
            logits = model(mels)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), correct / total

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Data Setup
    print("Setting up data...")
    train_dataset = MelDataset(CSV_TRAIN)
    val_dataset = MelDataset(CSV_VAL)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Model Setup
    model = get_passt_model(NUM_CLASSES)
    model = model.to(DEVICE)

    # 3. Optimizer & Loss
    # AdamW is standard for Transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    
    # Use CrossEntropy for single-label (one bird per file)
    # Use BCEWithLogitsLoss for multi-label (multiple birds possible)
    criterion = nn.CrossEntropyLoss() 

    # 4. Training Loop
    print(f"Starting training on {DEVICE}...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print("Saved Best Model!")
        
        print("-" * 30)

if __name__ == "__main__":
    main()