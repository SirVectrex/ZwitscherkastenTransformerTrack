import librosa
import numpy as np
import os
import sys
import contextlib
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# ==========================================
# KONFIGURATION
# ==========================================
INPUT_ROOT = Path("./audio_data")       
OUTPUT_ROOT = Path("./mel_spectrograms_uint8") # Neuer Ordnername zur Sicherheit
TARGET_SR = 32000                       
TARGET_DURATION = 10.0                  
N_MELS = 128                            

# QUANTISIERUNGS-PARAMETER
MIN_DB = -80.0  # Alles leiser als -80dB wird zu 0 (Stille)
MAX_DB = 0.0    # Referenz-Maximum (durch ref=np.max ist das Maximum immer 0)
# ==========================================

@contextlib.contextmanager
def ignore_stderr():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

def get_passt_spectrograms(file_path):
    try:
        with ignore_stderr():
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        return []

    if len(y) == 0: return []

    target_samples = int(TARGET_SR * TARGET_DURATION)
    spectrograms = []

    for start_idx in range(0, len(y), target_samples):
        end_idx = start_idx + target_samples
        chunk = y[start_idx:end_idx]
        
        if len(chunk) == 0: continue

        if len(chunk) < target_samples:
            n_repeats = int(np.ceil(target_samples / len(chunk)))
            y_padded = np.tile(chunk, n_repeats)
            chunk = y_padded[:target_samples]
        
        # Mel Berechnung
        mel_spec = librosa.feature.melspectrogram(
            y=chunk, 
            sr=sr, 
            n_fft=int(0.025 * TARGET_SR), 
            hop_length=int(0.010 * TARGET_SR), 
            n_mels=N_MELS
        )

        # Logarithmieren (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # --- NEU: QUANTISIERUNG ---
        # 1. Werte auf Bereich -80 bis 0 beschränken (Clipping)
        log_mel_spec = np.clip(log_mel_spec, MIN_DB, MAX_DB)
        
        # 2. Normalisieren auf 0.0 bis 1.0
        # (-80 wird zu 0.0, 0 wird zu 1.0)
        log_mel_spec = (log_mel_spec - MIN_DB) / (MAX_DB - MIN_DB)
        
        # 3. Skalieren auf 0-255 und als 8-Bit Integer speichern
        quantized_spec = (log_mel_spec * 255).astype(np.uint8)
        
        spectrograms.append(quantized_spec)
    
    return spectrograms

def process_file(job_info):
    src_path, rel_path = job_info
    
    # Pfad anpassen
    dst_base = OUTPUT_ROOT / rel_path.with_suffix('.npy')
    
    check_first = dst_base.with_name(f"{dst_base.stem}_0{dst_base.suffix}")
    if check_first.exists():
        return "Skipped"

    specs = get_passt_spectrograms(src_path)
    
    if specs:
        dst_base.parent.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(specs):
            save_path = dst_base.with_name(f"{dst_base.stem}_{i}{dst_base.suffix}")
            np.save(save_path, spec)
        return "Success"
    else:
        return "Error"

if __name__ == "__main__":
    num_workers = 8
    print(f"Suche MP3s in {INPUT_ROOT}...")
    all_files = list(INPUT_ROOT.rglob("*.mp3"))
    
    if not all_files:
        print("Keine Dateien gefunden!")
        exit()

    print(f"Verarbeite {len(all_files)} Dateien (Quantisierung AKTIV)...")
    
    jobs = []
    for f in all_files:
        jobs.append((f, f.relative_to(INPUT_ROOT)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_file, jobs), total=len(jobs), unit="file"))

    print("\nFertig! Die Dateien sind jetzt winzige uint8 Arrays.")