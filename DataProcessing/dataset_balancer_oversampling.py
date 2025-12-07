import os
import shutil
import random
import math

# --- KONFIGURATION ---
SOURCE_DIR = "./mel_spectrograms_uint8"
TARGET_ROOT = "./split_dataset_oversampling"

NUM_FOLDERS = 5  # "n": Auf wie viele Ordner sollen die Daten verteilt werden?
# ---------------------

def distribute_all_data(source, target_root, num_folders):
    if not os.path.exists(source):
        print(f"FEHLER: Quelle '{source}' nicht gefunden.")
        return

    # 1. Alle Klassen scannen und die GRÖSSTE Klasse finden
    classes = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    classes.sort()
    
    max_count = 0
    class_counts = {}

    print("Analysiere Datenbestand...")
    for class_name in classes:
        path = os.path.join(source, class_name)
        # Nur sichtbare Dateien zählen
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        count = len(files)
        class_counts[class_name] = count
        if count > max_count:
            max_count = count

    # Berechne, wie viele Bilder pro Ordner landen müssen, damit die größte Klasse aufgebraucht wird
    imgs_per_folder = max_count // num_folders
    
    print(f"Maximale Klasse hat {max_count} Bilder.")
    print(f"Bei {num_folders} Ordnern bedeutet das: {imgs_per_folder} Bilder pro Art je Ordner.")
    print(f"(Der Rest von {max_count % num_folders} Bildern wird ignoriert, damit es glatt aufgeht.)")
    print("-" * 50)

    # 2. Daten verteilen
    for class_name in classes:
        class_src_path = os.path.join(source, class_name)
        files = [f for f in os.listdir(class_src_path) if not f.startswith('.')]
        current_count = len(files)
        
        # Zielgröße für diese Klasse über alle Ordner hinweg
        total_needed = imgs_per_folder * num_folders
        
        balanced_files = []
        mode = ""

        # Wenn wir weniger haben als das Maximum -> AUFFÜLLEN (Oversampling)
        if current_count < total_needed:
            mode = "Oversampling"
            # Nimm Originale
            balanced_files = [(f, False) for f in files]
            # Berechne fehlende
            needed = total_needed - current_count
            # Ziehe Duplikate
            duplicates = random.choices(files, k=needed)
            balanced_files.extend([(f, True) for f in duplicates])
            
        # Wenn wir genau die Max-Klasse sind (oder mehr, was logisch nicht geht da max)
        else:
            mode = "Originalverteilung"
            # Wir nehmen exakt so viele, wie wir brauchen (schneiden evtl. Rest von Teilung ab)
            chosen = random.sample(files, total_needed)
            balanced_files = [(f, False) for f in chosen]

        # Mischen vor dem Verteilen
        random.shuffle(balanced_files)

        print(f"Verarbeite '{class_name}': {current_count} -> {total_needed} ({mode})")

        # In die n Ordner verteilen
        for i in range(num_folders):
            # Ordnernamen: subset_0, subset_1 ...
            split_folder_name = f"subset_{i}"
            split_class_path = os.path.join(target_root, split_folder_name, class_name)
            os.makedirs(split_class_path, exist_ok=True)

            # Slice berechnen
            start = i * imgs_per_folder
            end = start + imgs_per_folder
            batch = balanced_files[start:end]

            dup_counter = 0
            for file_name, is_duplicate in batch:
                src_file = os.path.join(class_src_path, file_name)
                
                if is_duplicate:
                    name, ext = os.path.splitext(file_name)
                    # Name ändern: _aug_subsetIndex_counter
                    new_name = f"{name}_aug_{i}_{dup_counter}{ext}"
                    dst_file = os.path.join(split_class_path, new_name)
                    dup_counter += 1
                else:
                    dst_file = os.path.join(split_class_path, file_name)
                
                shutil.copy2(src_file, dst_file)

    print("-" * 50)
    print("Fertig! Alle Daten der größten Klasse wurden aufgebraucht.")
    print("Kleinere Klassen wurden per Oversampling angeglichen.")

if __name__ == "__main__":
    distribute_all_data(SOURCE_DIR, TARGET_ROOT, NUM_FOLDERS)