import requests
import json
from pathlib import Path
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

# ==============================================================================
# ⚙️ KONFIGURATION & SCHALTER
# ==============================================================================

API_KEY = "93bfa6c3854af3b6f7429d1b36e5da0ac5491032"
CSV_FILENAME = "Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv"

# --- MAXIMALE DATEIGRÖSSE ---
# Nur Dateien herunterladen, die kleiner sind als X Megabyte.
MAX_FILE_SIZE_MB = 1  

# --- AUSWAHL VOGELARTEN (RANGE) ---
START_INDEX = 0      # Start bei Vogel Nr. X
END_INDEX   = 20     # Ende bei Vogel Nr. Y 

# --- DOWNLOAD MODUS ---
DOWNLOAD_MODE = "UNLIMITED" 
LIMIT_PER_CATEGORY = 30  

# Tuning
SEARCH_WORKERS = 4     
DOWNLOAD_WORKERS = 8  # Konservativer Wert für M5/Apple Silicon

# ==============================================================================


def load_species_from_csv(csv_path, start=0, end=None):
    try:
        df = pd.read_csv(csv_path, delimiter=';')
        if 'Wissenschaftlicher Name' not in df.columns:
            raise ValueError("Spalte 'Wissenschaftlicher Name' nicht gefunden!")
        
        if start < 0: start = 0
        if end is None:
            slice_df = df.iloc[start:]
        else:
            slice_df = df.iloc[start:end]
            
        species_list = slice_df['Wissenschaftlicher Name'].tolist()
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
    
    while page <= 50:
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
    return species_name, results


def search_unlimited(species_name):
    parts = species_name.split()
    query = f'gen:{parts[0]} sp:{parts[1]}' if len(parts) == 2 else f'gen:{parts[0]}'
    base_url = "https://xeno-canto.org/api/3/recordings"
    
    print(f"[{species_name}] Suche ALLES...", flush=True)
    collected = []
    page = 1
    
    while True:
        params = {"query": query, "key": API_KEY, "page": page}
        try:
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code != 200: break
            
            data = response.json()
            num_pages = int(data.get('numPages', 1))
            
            recordings = data.get('recordings', [])
            if not recordings: break
            
            for rec in recordings:
                if rec.get('q') in ['A', 'B', 'C']:
                    collected.append(clean_rec_data(rec))
            
            if page >= num_pages: break
            page += 1
            time.sleep(0.25)
        except Exception:
            break
            
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
    Lädt Datei nur herunter, wenn sie < MAX_FILE_SIZE_MB ist.
    """
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    
    try:
        # stream=True lädt nur die Header, nicht den Inhalt!
        with requests.get(rec['file_url'], stream=True, timeout=60) as response:
            if response.status_code == 200:
                
                # 1. Größe prüfen (Content-Length Header)
                total_length = response.headers.get('content-length')
                
                if total_length is not None:
                    if int(total_length) > max_bytes:
                        # Zu groß -> Ignorieren (gibt False zurück, aber keinen Fehler)
                        return False, 0
                
                # 2. Wenn Größe passt (oder unbekannt ist), laden wir den Inhalt
                filename = f"{rec['id']}_q{rec['quality']}.mp3"
                filepath = species_dir / filename
                
                downloaded_size = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Notbremse: Falls kein Header da war, aber Datei riesig wird
                            if downloaded_size > max_bytes:
                                f.close()
                                filepath.unlink() # Teilweise Datei löschen
                                return False, 0

                return True, downloaded_size
                
        return False, 0
    except Exception:
        return False, 0


# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================
if __name__ == "__main__":
    try:
        print("=" * 70)
        print(f"XENO-CANTO DOWNLOADER (Smart Filter)")
        print(f"Limit: Nur Dateien < {MAX_FILE_SIZE_MB} MB")
        print(f"Modus: {DOWNLOAD_MODE} | Worker: {DOWNLOAD_WORKERS}")
        print("=" * 70)

        SPECIES_LIST = load_species_from_csv(CSV_FILENAME, start=START_INDEX, end=END_INDEX)
        
        if not SPECIES_LIST: 
            print("Keine Vogelarten im gewählten Bereich gefunden.")
            sys.exit(0)

        # --- PHASE 1: SUCHE ---
        print("\n--- PHASE 1: Metadaten sammeln ---")
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
                except Exception: pass

        # --- PHASE 2: DOWNLOAD ---
        print("\n--- PHASE 2: Downloads (Smart Filter aktiv) ---")
        
        all_jobs = []
        metadata_files = list(Path("metadata").glob("*.json"))
        
        for meta_file in metadata_files:
            species_name = meta_file.stem.replace('_metadata', '')
            if species_name.replace('_', ' ') not in SPECIES_LIST:
                continue 

            dest_dir = Path("audio_data") / species_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            with open(meta_file, 'r', encoding='utf-8') as f:
                recs = json.load(f)
                for r in recs:
                    fname = f"{r['id']}_q{r['quality']}.mp3"
                    if not (dest_dir / fname).exists():
                        all_jobs.append((r, dest_dir))
        
        total_candidates = len(all_jobs)
        if total_candidates == 0:
            print("Nichts zu tun.")
            sys.exit(0)

        print(f"Prüfe {total_candidates} Kandidaten auf Größe...")
        
        completed = 0
        skipped_size = 0
        total_bytes = 0
        start_time_dl = time.time()
        
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = [executor.submit(download_single_file, job[0], job[1]) for job in all_jobs]
            
            for i, future in enumerate(as_completed(futures)):
                success, size_bytes = future.result()
                
                if success:
                    completed += 1
                    total_bytes += size_bytes
                else:
                    skipped_size += 1
                
                if i % 5 == 0 or i == total_candidates - 1:
                    elapsed = time.time() - start_time_dl
                    speed_mb_s = (total_bytes / 1024**2) / elapsed if elapsed > 0 else 0
                    percent = (i + 1) / total_candidates * 100
                    
                    print(f"Fortschritt: {i+1}/{total_candidates} ({percent:.1f}%) | "
                          f"Geladen: {completed} | Zu Groß/Fehler: {skipped_size} | "
                          f"Speed: {speed_mb_s:.2f} MB/s   ", end='\r')

        print(f"\n\nFERTIG! {completed} Dateien geladen. ({skipped_size} übersprungen)")

    except KeyboardInterrupt:
        print("\n\nABBRUCH DURCH BENUTZER.")
        os._exit(1)