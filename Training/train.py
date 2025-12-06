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
CSV_TRAIN = "./Training/train.csv"  # columns: filepath, label
CSV_VAL = "./Training/val.csv"
NUM_CLASSES = 60         # Change this to your number of bird species
BATCH_SIZE = 16          # PaSST is heavy; reduce this if you get CUDA OOM
LEARNING_RATE = 1e-5     # Low LR is best for finetuning
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"

def get_passt_model(num_classes):
    print("Loading PaSST model...")
    model = get_basic_model(mode="all", arch="passt_s_swa_p16_128_ap476")

    # 1. Bypass the Audio Frontend
    model.mel = nn.Identity()

    # 2. Replace the Classification Head
    # Check if the head is a Sequential block (LayerNorm + Linear)
    if isinstance(model.net.head, nn.Sequential):
        # The Linear layer is the last item in the list [-1]
        in_features = model.net.head[-1].in_features
        
        # Replace ONLY the final Linear layer (keeping the LayerNorm before it)
        model.net.head[-1] = nn.Linear(in_features, num_classes)
    else:
        # Fallback: It is just a single Linear layer
        in_features = model.net.head.in_features
        model.net.head = nn.Linear(in_features, num_classes)
    
    # Optional: Initialize weights for the new layer
    # torch.nn.init.xavier_uniform_(model.net.head[-1].weight)
    
    return model

# --- train.py - Corrected train_one_epoch function ---

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training")
    for mels, labels in loop:
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)
      
        print("Shape of mels:", mels.shape)
        # The data is already in [Batch, 1, 128, 998]

        optimizer.zero_grad()
        
        # Forward pass
        mels = mels.view(mels.size(0), 128, 998)
        
        # Forward pass
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

            mels = mels.view(mels.size(0), 128, 998)
            
            # Note: PaSST usually disables Patchout during eval automatically
            logits = model(mels)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), correct / total

def set_patchout_difficulty(model, difficulty="standard"):
    """
    Adjusts the structured Patchout (dropping time/freq strips).
    
    Standard PaSST (10s clips):
    - s_patchout_t: Drops time columns (vertical strips)
    - s_patchout_f: Drops freq rows (horizontal strips)
    """
    if difficulty == "hard":
        # Drop MORE information (Good if you have lots of data/overfitting)
        model.net.s_patchout_t = 40  # Default is usually around 20-30
        model.net.s_patchout_f = 6   # Default is usually 4
        print("Patchout set to HARD (High regularization)")
        
    elif difficulty == "light":
        # Drop LESS (Good for small datasets or very short/sparse bird calls)
        model.net.s_patchout_t = 10
        model.net.s_patchout_f = 2
        print("Patchout set to LIGHT (Low regularization)")
        
    else:
        # Restore defaults (approximate values for PaSST-S)
        model.net.s_patchout_t = 20
        model.net.s_patchout_f = 4
        print("Patchout set to STANDARD")

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

    #2.5 Optional: Adjust Patchout Difficulty
    set_patchout_difficulty(model, difficulty="standard")

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