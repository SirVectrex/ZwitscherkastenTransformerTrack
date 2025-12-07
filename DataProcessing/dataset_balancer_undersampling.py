import os
import shutil
import random

# --- KONFIGURATION ---
SOURCE_DIR = "./mel_spectrograms_uint8"
TARGET_ROOT = "./split_dataset_undersampled"

NUM_FOLDERS = 5  # "n": Auf wie viele Ordner verteilen?
# ---------------------

def undersample_split(source, target_root, num_folders):
    if not os.path.exists(source):
        print(f"FEHLER: Quelle '{source}' nicht gefunden.")
        return

    # 1. Alle Klassen scannen und das MINIMUM finden
    classes = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    classes.sort()
    
    min_count = float('inf')
    min_class_name = ""

    print("Analysiere Datenbestand...")
    
    # Vorher einmal durchlaufen, um das Minimum zu finden
    for class_name in classes:
        path = os.path.join(source, class_name)
        # Nur sichtbare Dateien zählen
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        count = len(files)
        
        if count > 0 and count < min_count:
            min_count = count
            min_class_name = class_name
        elif count == 0:
             print(f"Warnung: Klasse '{class_name}' ist leer!")

    if min_count == float('inf'):
        print("Fehler: Keine Dateien gefunden.")
        return

    # Berechne, wie viele Bilder pro Ordner landen
    # Beispiel: Minimum ist 82, n=4 Ordner -> 82 // 4 = 20 Bilder pro Ordner.
    # Die restlichen 2 werden verworfen, damit es glatt aufgeht.
    imgs_per_folder = min_count // num_folders

    print(f"Kleinste Klasse ist '{min_class_name}' mit {min_count} Bildern.")
    print(f"Wir reduzieren ALLE Klassen auf {imgs_per_folder * num_folders} Bilder.")
    print(f"Ergebnis: {num_folders} Ordner mit je {imgs_per_folder} Bildern pro Art.")
    
    if imgs_per_folder == 0:
        print("FEHLER: Die kleinste Klasse hat weniger Bilder als es Zielordner gibt!")
        return

    print("-" * 50)

    # 2. Daten verteilen
    for class_name in classes:
        class_src_path = os.path.join(source, class_name)
        files = [f for f in os.listdir(class_src_path) if not f.startswith('.')]
        
        # Zufällig mischen
        random.shuffle(files)
        
        # --- UNDERSAMPLING ---
        # Wir nehmen nur so viele, wie wir brauchen (basiert auf der kleinsten Klasse)
        total_needed = imgs_per_folder * num_folders
        selected_files = files[:total_needed]

        print(f"Verarbeite '{class_name}': {len(files)} -> {len(selected_files)} (Reduziert)")

        # In die n Ordner verteilen
        for i in range(num_folders):
            # Ordnernamen: subset_0, subset_1 ...
            split_folder_name = f"subset_{i}"
            split_class_path = os.path.join(target_root, split_folder_name, class_name)
            os.makedirs(split_class_path, exist_ok=True)

            # Slice berechnen (den "Kuchen" schneiden)
            start = i * imgs_per_folder
            end = start + imgs_per_folder
            batch = selected_files[start:end]

            for file_name in batch:
                src_file = os.path.join(class_src_path, file_name)
                dst_file = os.path.join(split_class_path, file_name)
                shutil.copy2(src_file, dst_file)

    print("-" * 50)
    print("Fertig! Das Dataset ist nun perfekt balanciert (basierend auf der kleinsten Klasse).")

if __name__ == "__main__":
    undersample_split(SOURCE_DIR, TARGET_ROOT, NUM_FOLDERS)