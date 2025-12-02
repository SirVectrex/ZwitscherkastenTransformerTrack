import os
from pathlib import Path

# ==========================================
# KONFIGURATION
# ==========================================
TARGET_FOLDER = Path("./audio_data")  # Wo liegen die Dateien?
MAX_SIZE_MB = 1.0                     # Alles darüber wird gelöscht
# ==========================================

def scan_and_clean():
    print(f"🔍 Scanne '{TARGET_FOLDER}' nach Dateien größer als {MAX_SIZE_MB} MB...")
    
    if not TARGET_FOLDER.exists():
        print("❌ Ordner nicht gefunden!")
        return

    files_to_delete = []
    total_size_bytes = 0
    limit_bytes = MAX_SIZE_MB * 1024 * 1024

    # 1. Alle Dateien finden
    # rglob('*') sucht rekursiv in allen Unterordnern
    for file_path in TARGET_FOLDER.rglob("*"):
        if file_path.is_file():
            try:
                size = file_path.stat().st_size
                if size > limit_bytes:
                    files_to_delete.append((file_path, size))
                    total_size_bytes += size
            except Exception as e:
                print(f"Fehler beim Lesen von {file_path}: {e}")

    count = len(files_to_delete)
    
    if count == 0:
        print("✅ Keine zu großen Dateien gefunden. Alles sauber!")
        return

    # 2. Zusammenfassung anzeigen
    total_size_mb = total_size_bytes / (1024 * 1024)
    print("\n" + "="*40)
    print(f"⚠️  GEFUNDENE KANDIDATEN: {count} Dateien")
    print(f"💾 Speicherplatz freizugeben: {total_size_mb:.2f} MB")
    print("="*40)
    
    # Die ersten 5 Beispiele zeigen
    print("Beispiele:")
    for f, s in files_to_delete[:5]:
        print(f" - {f.name} ({s/(1024*1024):.2f} MB)")
    if count > 5:
        print(f" ... und {count - 5} weitere.")

    # 3. Sicherheitsabfrage
    print("\nWillst du diese Dateien wirklich LÖSCHEN?")
    choice = input("Tippe 'ja' zum Löschen (alles andere bricht ab): ").strip().lower()

    if choice == 'ja':
        print("\n🗑️  Lösche Dateien...")
        deleted_count = 0
        
        for file_path, _ in files_to_delete:
            try:
                file_path.unlink() # Löscht die Datei
                deleted_count += 1
            except Exception as e:
                print(f"Konnte {file_path.name} nicht löschen: {e}")
        
        print(f"✅ Fertig! {deleted_count} Dateien gelöscht.")
    else:
        print("❌ Abbruch. Nichts wurde gelöscht.")

if __name__ == "__main__":
    scan_and_clean()