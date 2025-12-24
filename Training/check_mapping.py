from dataset import MelDataset
import pandas as pd

# Pfade anpassen
CSV_TRAIN = "./Training/train.csv"
CSV_VAL = "./Training/val.csv"
STATS_FILE = "/dev/shm/schoen/output/normalization_stats.json"

print("--- Prüfe Zuordnungen ---")

# 1. Train laden
ds_train = MelDataset(CSV_TRAIN, stats_file=STATS_FILE)
# Wir holen uns die interne Liste (meistens .classes oder wir bauen sie nach)
if hasattr(ds_train, 'classes'):
    classes_train = ds_train.classes
else:
    # Falls dein Dataset keine .classes speichert, rekonstruieren wir es wie das Dataset es tut
    df = pd.read_csv(CSV_TRAIN)
    classes_train = sorted(df['label'].unique())

# 2. Val laden
ds_val = MelDataset(CSV_VAL, stats_file=STATS_FILE)
if hasattr(ds_val, 'classes'):
    classes_val = ds_val.classes
else:
    df = pd.read_csv(CSV_VAL)
    classes_val = sorted(df['label'].unique())

print(f"Anzahl Klassen Train: {len(classes_train)}")
print(f"Anzahl Klassen Val:   {len(classes_val)}")

print("\n--- Stichproben-Vergleich ---")
# Wir prüfen die ersten 5 Indizes
limit = min(len(classes_train), len(classes_val), 10)
mismatch_found = False

for i in range(limit):
    vogel_train = classes_train[i]
    vogel_val = classes_val[i]
    
    status = "OK" if vogel_train == vogel_val else "FEHLER/VERSCHIEBUNG!"
    if vogel_train != vogel_val:
        mismatch_found = True
        
    print(f"Index {i}: Train='{vogel_train}'  vs.  Val='{vogel_val}' -> {status}")

if mismatch_found or len(classes_train) != len(classes_val):
    print("\n ALARM: Die Zuordnungen stimmen nicht überein!")
    print(" Dein Val-Loss und deine Accuracy sind FALSCH.")
else:
    print("\n Alles gut. Die Listen sind synchron.")
