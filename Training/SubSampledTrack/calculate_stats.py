import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

# --- EINSTELLUNGEN ---
# Dein Pfad im RAM
DATA_ROOT = Path("/dev/shm/schoen/data_unzipped/split_dataset_1000_preprocessed")
SAMPLE_SIZE = 5000  # 5000 Dateien reichen für eine sehr genaue Schätzung

def main():
    print(f"--- START: Statistik-Berechnung ---")
    print(f"Suche Dateien in: {DATA_ROOT}")

    if not DATA_ROOT.exists():
        print("❌ FEHLER: Pfad nicht gefunden!")
        return

    # 1. Alle .npy Dateien finden (egal in welchem Unterordner)
    # rglob sucht rekursiv in allen Unterordnern
    all_files = list(DATA_ROOT.rglob("*.npy"))
    
    total_files = len(all_files)
    print(f"Gefundene Dateien: {total_files}")

    if total_files == 0:
        print("❌ Keine .npy Dateien gefunden.")
        return

    # 2. Stichprobe ziehen
    if total_files > SAMPLE_SIZE:
        print(f"Nehme zufällige Stichprobe von {SAMPLE_SIZE} Dateien...")
        files_to_process = random.sample(all_files, SAMPLE_SIZE)
    else:
        files_to_process = all_files

    # 3. Daten laden
    collected_values = []
    
    print("Lade Daten und extrahiere Werte...")
    for file_path in tqdm(files_to_process):
        try:
            # Datei laden
            mel_spec = np.load(file_path)
            
            # Wir brauchen nur eine Zufallsauswahl der Pixel aus jedem Bild, 
            # um Speicher zu sparen, oder wir nehmen das ganze Bild.
            # Bei 5000 Dateien x (128*998 floats) wird der RAM voll.
            # Trick: Wir berechnen Mean/Std pro Datei und mitteln dann (mathematisch nicht 100% exakt, aber für Normalisierung ok)
            # ODER: Wir nehmen nur jeden 10. Wert (Subsampling), das ist statistisch sauber.
            
            # Flatten und Subsampling (jedes 10. Element), um RAM zu sparen
            flat_data = mel_spec.flatten()[::10]
            collected_values.append(flat_data)
            
        except Exception as e:
            print(f"Warnung bei Datei {file_path}: {e}")

    # 4. Alles zusammenfügen
    print("Konkateniere Daten...")
    big_array = np.concatenate(collected_values)
    
    # 5. Berechnen
    print("Berechne Mean und Std...")
    mean_val = np.mean(big_array)
    std_val = np.std(big_array)

    print("\n" + "="*40)
    print("ERGEBNIS (Kopiere diese Werte!)")
    print("="*40)
    print(f"mean = {mean_val:.7f}")
    print(f"std  = {std_val:.7f}")
    print("="*40)
    
    # JSON Format für Copy-Paste
    print("\nAls JSON Block für deine Datei:")
    print("passt_stats = {")
    print(f'    "mean": {mean_val:.7f},')
    print(f'    "std": {std_val:.7f}')
    print("}")

if __name__ == "__main__":
    main()
