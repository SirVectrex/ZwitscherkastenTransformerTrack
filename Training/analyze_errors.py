import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# --- DEINE IMPORTS ---
from dataset import MelDataset
from hear21passt.base import get_basic_model

# --- KONFIGURATION ---
# Wir müssen hier 64 lassen, damit der Checkpoint geladen werden kann!
NUM_CLASSES = 64  
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pfade
CHECKPOINT_PATH = "/dev/shm/schoen/checkpoints/best_model.pth"
CSV_TRAIN = "./Training/train.csv"
CSV_VAL = "./Training/val.csv"
STATS_FILE = "/dev/shm/schoen/output/normalization_stats.json"

def load_model(num_classes):
    print(f"Lade Modell mit {num_classes} Ausgängen...")
    model = get_basic_model(mode="all", arch="passt_s_swa_p16_128_ap476")
    model.mel = nn.Identity()
    
    if isinstance(model.net.head, nn.Sequential):
        in_features = model.net.head[-1].in_features
        model.net.head[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.net.head.in_features
        model.net.head = nn.Linear(in_features, num_classes)
        
    return model

def get_class_names(csv_path):
    """Liest die Vogelnamen (oder Nummern) aus der CSV."""
    df = pd.read_csv(csv_path)
    # Sortieren ist wichtig, damit Index 0 auch Klasse 0 ist
    classes = sorted(df['label'].unique())
    # Alles in Strings umwandeln, damit es in der Tabelle schön aussieht
    return [str(c) for c in classes]

def analyze():
    print("--- STARTE ANALYSE ---")
    
    # 1. Daten laden
    val_dataset = MelDataset(CSV_VAL, stats_file=STATS_FILE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Namen laden und auffüllen (Der WICHTIGE Teil)
    print(f"Lade Klassennamen aus {CSV_TRAIN}...")
    class_names = get_class_names(CSV_TRAIN)
    num_found = len(class_names)
    print(f"Gefunden in CSV: {num_found} Klassen.")
    
    # --- FIX FÜR 59 vs 64 ---
    if num_found < NUM_CLASSES:
        diff = NUM_CLASSES - num_found
        print(f"ACHTUNG: Modell erwartet {NUM_CLASSES}, Daten haben nur {num_found}.")
        print(f"-> Fülle {diff} leere Plätze auf (Padding).")
        for i in range(diff):
            class_names.append(f"Leerstand_{num_found + i}")
    # ------------------------

    # 3. Modell laden
    model = load_model(NUM_CLASSES).to(DEVICE)
    print(f"Lade Checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    state_dict = checkpoint.get("model_state", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Info: Strict loading failed, trying sloppy load... ({e})")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()

    # 4. Vorhersagen sammeln
    all_preds = []
    all_labels = []

    print("Berechne Vorhersagen...")
    with torch.no_grad():
        for mels, labels in tqdm(val_loader):
            mels = mels.to(DEVICE)
            if mels.dim() == 4: mels = mels.squeeze(1)
            
            logits = model(mels)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Auswertung
    print("\n" + "="*40)
    print("ERGEBNISSE")
    print("="*40)

    # Wir erzwingen eine Matrix der Größe NUM_CLASSES x NUM_CLASSES (64x64)
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    
    # Genauigkeit berechnen
    with np.errstate(divide='ignore', invalid='ignore'):
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_accuracies = np.nan_to_num(class_accuracies)

    # DataFrame erstellen - Jetzt sind beide Listen garantiert gleich lang!
    acc_df = pd.DataFrame({
        'Vogel_ID': class_names,
        'Genauigkeit': class_accuracies,
        'Anzahl_Samples': cm.sum(axis=1)
    }).sort_values('Genauigkeit')

    # Nur Vögel anzeigen, die auch wirklich getestet wurden (nicht die leeren Plätze)
    active = acc_df[acc_df['Anzahl_Samples'] > 0]

    print("\n--- Die 5 SCHLECHTESTEN Vögel ---")
    print(active.head(5).to_string(index=False))

    print("\n--- Die 5 BESTEN Vögel ---")
    print(active.tail(5).to_string(index=False))

    # Verwechslungen
    np.fill_diagonal(cm, 0)
    indices = np.argsort(cm.flatten())[::-1][:10]
    
    print("\n--- TOP 10 VERWECHSLUNGEN ---")
    print(f"{'Wahrheit':<20} verwechselt mit -> {'Vorhersage':<20} (Anzahl)")
    print("-" * 60)
    
    for idx in indices:
        true_idx = idx // NUM_CLASSES
        pred_idx = idx % NUM_CLASSES
        count = cm[true_idx, pred_idx]
        
        if count > 0:
            name_wahr = class_names[true_idx]
            name_falsch = class_names[pred_idx]
            print(f"{name_wahr:<20} -> {name_falsch:<20} : {count}x")

    # Grafik
    try:
        if cm.max() > 0:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Reds')
            plt.title(f'Confusion Matrix ({num_found} Klassen + Padding)')
            plt.tight_layout()
            plt.savefig("confusion_matrix_errors.png")
            print("\nGrafik gespeichert: confusion_matrix_errors.png")
    except Exception:
        pass

if __name__ == "__main__":
    analyze()
