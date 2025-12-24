import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio.transforms as T
import torch.nn.functional as F

# --- DEINE IMPORTS ---
from dataset import MelDataset
from hear21passt.base import get_basic_model

# --- KONFIGURATION ---
CHECKPOINT_PATH = "/dev/shm/schoen/checkpoints/best_model.pth"
CSV_VAL = "./Training/val.csv"
STATS_FILE = "/dev/shm/schoen/output/normalization_stats.json"

NUM_CLASSES = 59 
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globale Transforms
tta_freq_mask = T.FrequencyMasking(freq_mask_param=20).to(DEVICE)
tta_time_mask = T.TimeMasking(time_mask_param=50).to(DEVICE)

def get_model(num_classes):
    print(f"--> Initialisiere PaSST für {num_classes} Klassen...")
    
    # WICHTIG: Wir nutzen mode="all", weil der Wrapper sonst abstürzt.
    # Das bedeutet, der Output wird 827 groß sein (59 Klassen + 768 Embeddings).
    # Wir schneiden den Rest später einfach ab.
    model = get_basic_model(mode="all", arch="passt_s_swa_p16_128_ap476")
    model.mel = nn.Identity() 
    
    if hasattr(model.net, "head"):
        in_features = 768 
        model.net.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes)
        )
    return model

def apply_tta_augs(mels):
    """
    Erzeugt 4 Varianten für TTA.
    Input ist 3D: [Batch, Freq, Time]
    """
    variants = []
    # 1. Original
    variants.append(mels)
    # 2. Freq Maske
    variants.append(tta_freq_mask(mels.clone()))
    # 3. Time Maske
    variants.append(tta_time_mask(mels.clone()))
    # 4. Shift
    shift = 100
    shifted = torch.zeros_like(mels)
    shifted[:, :, shift:] = mels[:, :, :-shift]
    variants.append(shifted)
    return variants

def main():
    print(f"--- STARTE TTA VALIDIERUNG ({NUM_CLASSES} Klassen) ---")
    
    val_dataset = MelDataset(CSV_VAL, stats_file=STATS_FILE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = get_model(NUM_CLASSES).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Lade Checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint

        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Checkpoint erfolgreich (strict) geladen.")
        except RuntimeError as e:
            print(f"⚠️ Warnung (normal bei PaSST): {e}")
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"❌ FEHLER: Checkpoint nicht gefunden.")
        return

    model.eval()
    
    correct = 0
    total = 0
    
    print("Berechne Vorhersagen...")
    first_batch = True

    with torch.no_grad():
        for mels, labels in tqdm(val_loader):
            mels = mels.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # --- INPUT FIX (3D erzwingen) ---
            B = mels.shape[0]
            F_dim = mels.shape[-2] # 128
            T_dim = mels.shape[-1] # 998
            mels = mels.contiguous().view(B, F_dim, T_dim)

            if first_batch:
                print(f"[DEBUG] Input Shape: {mels.shape}")
                first_batch = False

            # --- TTA START ---
            batch_variants = apply_tta_augs(mels) 
            summed_probs = torch.zeros((B, NUM_CLASSES)).to(DEVICE)
            
            for variant in batch_variants:
                # Output hier ist [Batch, 827] wegen mode="all"
                raw_output = model(variant)
                
                # --- OUTPUT SLICING (Der Metzger-Fix 🔪) ---
                # Wir nehmen nur die ersten 59 Spalten. Das sind unsere Klassen.
                # Der Rest (Spalte 59 bis 826) sind Embeddings, die werfen wir weg.
                logits = raw_output[:, :NUM_CLASSES]
                
                probs = F.softmax(logits, dim=1)
                summed_probs += probs
                
            _, predicted = torch.max(summed_probs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = correct / total
    
    print("\n" + "="*40)
    print(f"TTA ERGEBNIS")
    print("="*40)
    print(f"Getestete Vögel: {total}")
    print(f"TTA Accuracy:    {acc:.4%}")
    print("="*40)

if __name__ == "__main__":
    main()
