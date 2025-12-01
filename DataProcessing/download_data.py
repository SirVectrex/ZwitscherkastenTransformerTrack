import requests
import json
from pathlib import Path
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# ⚙️ KONFIGURATION & SCHALTER
# ==============================================================================

API_KEY = "93bfa6c3854af3b6f7429d1b36e5da0ac5491032"
CSV_FILENAME = "Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv"

# --- AUSWAHL VOGELARTEN (RANGE) ---
# Python nutzt 0-basierte Indexierung.
# Beispiel: 5 bis 9 lädt die Zeilen 5, 6, 7, 8 (also 4 Stück).
START_INDEX = 5      # Start bei Vogel Nr. X (0 = der erste in der CSV)
END_INDEX   = 9      # Ende bei Vogel Nr. Y (None = bis zum Ende der Datei)

# --- DOWNLOAD MODUS ---
# "LIMITED"   = Lädt exakt die Anzahl in 'LIMIT_PER_CATEGORY' pro Qualität (A, B, C)
# "UNLIMITED" = Lädt ALLES verfügbare
DOWNLOAD_MODE = "LIMITED" 

# Nur relevant für "LIMITED":
LIMIT_PER_CATEGORY = 30  

# Tuning
SEARCH_WORKERS = 4     
DOWNLOAD_WORKERS = 20  

# ==============================================================================


def load_species_from_csv(csv_path, start=0, end=None):
    """Liest die CSV und holt einen definierten Bereich (Slice)."""
    try:
        df = pd.read_csv(csv_path, delimiter=';')
        if 'Wissenschaftlicher Name' not in df.columns:
            raise ValueError("Spalte 'Wissenschaftlicher Name' nicht gefunden!")
        
        # Sicherstellen, dass Start nicht negativ ist
        if start < 0: start = 0
        
        # Den Bereich (Slice) auswählen
        # df.iloc[start:end] schneidet die Zeilen aus dem DataFrame
        if end is None:
            slice_df = df.iloc[start:]
        else:
            slice_df = df.iloc[start:end]
            
        species_list = slice_df['Wissenschaftlicher Name'].tolist()
            
        # Bereinigen (Leerzeichen entfernen)
        species_list = [name.strip() for name in species_list if isinstance(name, str) and name.strip()]
        
        print(f"Info: Lade Zeilen {start} bis {end if end else 'Ende'} (Total: {len(species_list)} Arten)")
        return species_list
    except Exception as e:
        print(f"Fehler beim Lesen der CSV: {e}")
        return []


def search_limited(species_name, target_count):
    parts = species_name.split()
    query = f'gen:{parts[0]} sp:{parts[1]}' if len(parts) == 2 else f'gen:{parts[0]}'
    base_url = "https://xeno-canto.org/api/3/recordings"
    
    print(f"[{species_name}] Suche (Ziel: je {target_count} A/B/C)...", flush=True)
    
    collected = {'A': [], 'B': [], 'C': []}
    page = 1
    max_pages = 50 
    
    while page <= max_pages:
        params = {"query": query, "key": API_KEY, "page": page}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code != 200: break
            
            data = response.json()
            recordings = data.get('recordings', [])
            if not recordings: break
            
            for rec in recordings:
                q = rec.get('q', 'E')
                if q in ['A', 'B', 'C'] and len(collected[q]) < target_count:
                    collected[q].append(clean_rec_data(rec))
            
            if all(len(collected[q]) >= target_count for q in ['A', 'B', 'C']):
                break
            
            page += 1
            time.sleep(0.2)
        except Exception:
            break
            
    results = collected['A'] + collected['B'] + collected['C']
    print(f"[{species_name}] ✓ Gefunden: {len(results)} (Limitiert)", flush=True)
    return species_name, results


def search_unlimited(species_name):
    parts = species_name.split()
    query = f'gen:{parts[0]} sp:{parts[1]}' if len(parts) == 2 else f'gen:{parts[0]}'
    base_url = "https://xeno-canto.org/api/3/recordings"
    
    print(f"[{species_name}] Suche ALLES...", flush=True)
    
    collected = []
    page = 1
    total_pages = 1 
    
    while page <= total_pages:
        params = {"query": query, "key": API_KEY, "page": page}
        try:
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code != 200: break
            
            data = response.json()
            
            if page == 1:
                total_pages = int(data.get('numPages', 1))
                if total_pages > 5:
                    print(f"[{species_name}] -> {total_pages} Seiten gefunden.", flush=True)

            recordings = data.get('recordings', [])
            if not recordings: break
            
            for rec in recordings:
                q = rec.get('q', 'E')
                if q in ['A', 'B', 'C']:
                    collected.append(clean_rec_data(rec))
            
            page += 1
            time.sleep(0.25)
            
        except Exception as e:
            print(f"[{species_name}] Fehler Seite {page}: {e}")
            break
            
    print(f"[{species_name}] ✓ FERTIG: {len(collected)} Aufnahmen.", flush=True)
    return species_name, collected


def clean_rec_data(rec):
    return {
        'id': rec['id'], 'quality': rec.get('q'), 'length': rec['length'],
        'file_url': rec['file'], 'country': rec['cnt'],
        'location': rec['loc'], 'date': rec['date']
    }


def save_metadata_wrapper(species_name, recordings):
    if not recordings: return
    output_dir = "metadata"
    Path(output_dir).mkdir(exist_ok=True)
    filename = f"{species_name.replace(' ', '_')}_metadata.json"
    filepath = Path(output_dir) / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(recordings, f, indent=2, ensure_ascii=False)


def download_single_file(rec, species_dir):
    """
    Lädt Datei herunter.
    Rückgabe: (Erfolg: bool, Bytes: int)
    """
    try:
        response = requests.get(rec['file_url'], timeout=60) 
        if response.status_code == 200:
            filename = f"{rec['id']}_q{rec['quality']}.mp3"
            filepath = species_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            # WICHTIG: Wir geben die Größe in Bytes zurück
            return True, len(response.content)
        return False, 0
    except Exception:
        return False, 0


# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================
if __name__ == "__main__":
    
    print("=" * 70)
    print(f"XENO-CANTO DOWNLOADER (Range Mode)")
    print(f"Modus: {DOWNLOAD_MODE} | Worker: {DOWNLOAD_WORKERS}")
    print(f"Lade CSV Bereich: Index {START_INDEX} bis {END_INDEX}")
    print("=" * 70)

    # 1. Liste laden (mit Range)
    SPECIES_LIST = load_species_from_csv(CSV_FILENAME, start=START_INDEX, end=END_INDEX)
    
    if not SPECIES_LIST: 
        print("Keine Vogelarten im gewählten Bereich gefunden.")
        exit()

    print(f"Ausgewählte Arten: {SPECIES_LIST}")
    time.sleep(2)

    # --- PHASE 1: SUCHE ---
    print("\n--- PHASE 1: Metadaten sammeln ---")
    start_time_search = time.time()
    
    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as executor:
        future_to_species = {}
        for species in SPECIES_LIST:
            if DOWNLOAD_MODE == "UNLIMITED":
                future = executor.submit(search_unlimited, species)
            else:
                future = executor.submit(search_limited, species, LIMIT_PER_CATEGORY)
            future_to_species[future] = species
        
        for future in as_completed(future_to_species):
            try:
                name, recs = future.result()
                save_metadata_wrapper(name, recs)
            except Exception as e:
                pass

    print(f"\nSuche fertig in {time.time() - start_time_search:.1f}s.")

    # --- PHASE 2: DOWNLOAD ---
    print("\n--- PHASE 2: Downloads ---")
    
    all_jobs = []
    metadata_files = list(Path("metadata").glob("*.json"))
    
    for meta_file in metadata_files:
        species_name = meta_file.stem.replace('_metadata', '')
        
        # WICHTIG: Wenn wir Ranges nutzen, wollen wir oft NUR die gerade gesuchten Arten laden
        # und nicht alles, was zufällig noch im metadata Ordner liegt.
        # Check: Ist dieser Vogel in unserer aktuellen Liste?
        original_name_clean = species_name.replace('_', ' ')
        if original_name_clean not in SPECIES_LIST:
            # Optional: Wenn du das nicht willst (also ALLES im Ordner laden willst),
            # kommentiere die nächsten 2 Zeilen aus.
            continue 

        dest_dir = Path("audio_data") / species_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            recs = json.load(f)
            for r in recs:
                fname = f"{r['id']}_q{r['quality']}.mp3"
                if not (dest_dir / fname).exists():
                    all_jobs.append((r, dest_dir))
    
    total = len(all_jobs)
    if total == 0:
        print("Alle Dateien im gewählten Bereich bereits vorhanden.")
        exit()

    print(f"Lade {total} fehlende Dateien für {len(SPECIES_LIST)} Arten...")
    
    completed = 0
    total_bytes = 0
    start_time_dl = time.time()
    
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(download_single_file, job[0], job[1]) for job in all_jobs]
        
        for i, future in enumerate(as_completed(futures)):
            success, size_bytes = future.result()
            
            if success:
                completed += 1
                total_bytes += size_bytes
            
            # --- GESCHWINDIGKEITS-ANZEIGE ---
            if i % 5 == 0 or i == total - 1:
                elapsed = time.time() - start_time_dl
                if elapsed > 0:
                    mb_downloaded = total_bytes / (1024 * 1024)
                    speed_mb_s = mb_downloaded / elapsed
                    percent = (i + 1) / total * 100
                    
                    print(f"Status: {completed}/{total} ({percent:.1f}%) | "
                          f"Gesamt: {mb_downloaded:.1f} MB | "
                          f"Speed: {speed_mb_s:.2f} MB/s   ", end='\r')

    print(f"\n\nFERTIG! {completed} Dateien geladen. Gesamtgröße: {total_bytes / (1024*1024):.1f} MB")