import requests
import json
from pathlib import Path
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

# ==============================================================================
# ⚙️ KONFIGURATION
# ==============================================================================

API_KEY = "93bfa6c3854af3b6f7429d1b36e5da0ac5491032"
CSV_FILENAME = "Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv"

# --- SMART FILTER (DAUER) ---
# Ignoriere Aufnahmen > 45 Sekunden (spart Zeit & Speicher)
MAX_DURATION_SECONDS = 60 

# --- RANGE & MODUS ---
START_INDEX = 22      
END_INDEX   = 32     
DOWNLOAD_MODE = "UNLIMITED" 
LIMIT_PER_CATEGORY = 30 

# --- TUNING ---
SEARCH_WORKERS = 4     
DOWNLOAD_WORKERS = 8  

# ==============================================================================

def load_species_from_csv(csv_path, start=0, end=None):
    try:
        df = pd.read_csv(csv_path, delimiter=';')
        if 'Wissenschaftlicher Name' not in df.columns:
            raise ValueError("Spalte 'Wissenschaftlicher Name' nicht gefunden!")
        
        if start < 0: start = 0
        if end is None: slice_df = df.iloc[start:]
        else: slice_df = df.iloc[start:end]
            
        species_list = slice_df['Wissenschaftlicher Name'].tolist()
        return [name.strip() for name in species_list if isinstance(name, str) and name.strip()]
    except Exception as e:
        print(f"CSV Fehler: {e}")
        return []

def parse_duration(duration_str):
    try:
        if ':' in str(duration_str):
            parts = str(duration_str).split(':')
            return int(parts[0]) * 60 + float(parts[1])
        return float(duration_str)
    except:
        return 0.0

def clean_rec_data(rec):
    return {
        'id': rec['id'], 
        'quality': rec.get('q'), 
        'length': rec.get('length'),
        'file_url': rec['file']
    }

def search_unlimited(species_name):
    parts = species_name.split()
    query = f'gen:{parts[0]} sp:{parts[1]}' if len(parts) == 2 else f'gen:{parts[0]}'
    base_url = "https://xeno-canto.org/api/3/recordings"
    print(f"[{species_name}] Suche...", flush=True)
    collected = []
    page = 1
    
    while True:
        try:
            r = requests.get(base_url, params={"query": query, "key": API_KEY, "page": page}, timeout=15)
            if r.status_code != 200: break
            data = r.json()
            if not data.get('recordings'): break
            
            for rec in data['recordings']:
                if rec.get('q') not in ['A', 'B', 'C']: continue
                
                # Metadaten-Filter
                if parse_duration(rec.get('length', '0')) > MAX_DURATION_SECONDS:
                    continue

                collected.append(clean_rec_data(rec))
            
            if page >= int(data.get('numPages', 1)): break
            page += 1
            time.sleep(0.2)
        except: break
    return species_name, collected

def search_limited(species_name, limit):
    parts = species_name.split()
    query = f'gen:{parts[0]} sp:{parts[1]}' if len(parts) == 2 else f'gen:{parts[0]}'
    base_url = "https://xeno-canto.org/api/3/recordings"
    print(f"[{species_name}] Suche (Limit {limit})...", flush=True)
    collected = {'A': [], 'B': [], 'C': []}
    page = 1
    
    while page <= 50:
        try:
            r = requests.get(base_url, params={"query": query, "key": API_KEY, "page": page}, timeout=10)
            if r.status_code != 200: break
            data = r.json()
            if not data.get('recordings'): break
            
            for rec in data['recordings']:
                q = rec.get('q', 'E')
                if q not in ['A','B','C']: continue
                if parse_duration(rec.get('length', '0')) > MAX_DURATION_SECONDS: continue

                if len(collected[q]) < limit:
                    collected[q].append(clean_rec_data(rec))
            
            if all(len(collected[k]) >= limit for k in ['A','B','C']): break
            page += 1
            time.sleep(0.2)
        except: break
    return species_name, collected['A'] + collected['B'] + collected['C']

def save_metadata_wrapper(species_name, recordings):
    if not recordings: return
    Path("metadata").mkdir(exist_ok=True)
    fname = Path("metadata") / f"{species_name.replace(' ', '_')}_metadata.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(recordings, f, indent=2)

def download_single_file(rec, species_dir):
    """
    Lädt Datei und gibt (Erfolg, Größe_in_Bytes) zurück.
    """
    try:
        # Stream nutzen, um Größe sauber zu messen
        with requests.get(rec['file_url'], stream=True, timeout=60) as r:
            if r.status_code == 200:
                filename = f"{rec['id']}_q{rec['quality']}.mp3"
                filepath = species_dir / filename
                
                downloaded = 0
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                return True, downloaded
            return False, 0
    except:
        return False, 0

# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================
if __name__ == "__main__":
    try:
        print("=" * 60)
        print(f"XENO-CANTO DOWNLOADER (Speed Display)")
        print(f"Filter: <= {MAX_DURATION_SECONDS}s | Modus: {DOWNLOAD_MODE}")
        print("=" * 60)

        SPECIES = load_species_from_csv(CSV_FILENAME, START_INDEX, END_INDEX)
        if not SPECIES: sys.exit(0)

        # PHASE 1: Metadaten
        """
        print("\n--- Metadaten sammeln ---")
        with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as exc:
            futs = {}
            for s in SPECIES:
                if DOWNLOAD_MODE == "UNLIMITED": fut = exc.submit(search_unlimited, s)
                else: fut = exc.submit(search_limited, s, LIMIT_PER_CATEGORY)
                futs[fut] = s
            for f in as_completed(futs):
                try: n, r = f.result(); save_metadata_wrapper(n, r)
                except: pass
        """

        # PHASE 2: Downloads
        print("\n--- Downloads ---")
        jobs = []
        for meta in Path("metadata").glob("*.json"):
            name = meta.stem.replace('_metadata', '').replace('_', ' ')
            if name not in SPECIES: continue
            
            dest = Path("audio_data") / meta.stem.replace('_metadata', '')
            dest.mkdir(parents=True, exist_ok=True)
            
            with open(meta) as f:
                for r in json.load(f):
                    if not (dest / f"{r['id']}_q{r['quality']}.mp3").exists():
                        jobs.append((r, dest))
        
        total = len(jobs)
        if total == 0: print("Alles erledigt."); sys.exit(0)
        
        print(f"Lade {total} gefilterte Dateien...")
        
        # Variablen für Speed-Anzeige
        done = 0
        total_bytes = 0
        start_time_dl = time.time()
        
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as exc:
            futures = [exc.submit(download_single_file, j[0], j[1]) for j in jobs]
            
            for i, f in enumerate(as_completed(futures)):
                success, size = f.result()
                
                if success:
                    done += 1
                    total_bytes += size
                
                # Anzeige aktualisieren (alle 5 Dateien oder am Ende)
                if i % 5 == 0 or i == total - 1:
                    pct = (i+1)/total * 100
                    
                    # Speed berechnen
                    elapsed = time.time() - start_time_dl
                    if elapsed > 0:
                        mb = total_bytes / (1024 * 1024)
                        speed = mb / elapsed
                    else:
                        speed = 0.0
                    
                    print(f"Status: {done}/{total} ({pct:.1f}%) | "
                          f"Gesamt: {mb:.1f} MB | "
                          f"Speed: {speed:.2f} MB/s   ", end='\r')

        print(f"\n\nFertig! {done} Dateien geladen.")

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
        os._exit(1)