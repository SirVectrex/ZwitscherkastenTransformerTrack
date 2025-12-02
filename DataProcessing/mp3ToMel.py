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
OUTPUT_ROOT = Path("./mel_spectrograms") 
TARGET_SR = 32000                       
TARGET_DURATION = 10.0                  
N_MELS = 128                            
# ==========================================

# --- HILFSFUNKTION: C-LEVEL WARNUNGEN UNTERDRÜCKEN ---
@contextlib.contextmanager
def ignore_stderr():
    """
    Leitet stderr temporär ins Leere um. 
    Stoppt nervige 'Xing stream' und 'Header' Warnungen von C-Bibliotheken.
    """
    try:
        # Öffne das "Nichts" (Null Device)
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Speichere den aktuellen stderr Kanal (Kanal 2)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        # Leite Kanal 2 auf devnull um
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except Exception:
        # Falls das System das Umleiten nicht erlaubt, mach einfach weiter
        yield
    finally:
        # Stelle den alten stderr Kanal wieder her
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

def get_passt_spectrograms(file_path):
    """
    Lädt MP3 (leise), splittet in 10s Segmente, wendet Repeat-Padding an
    und berechnet Mel-Spektrogramme.
    """
    try:
        # [cite_start]1. Laden & Resampling (32kHz Mono) [cite: 142]
        # Wir wickeln das Laden in unseren "Schalldämpfer"
        with ignore_stderr():
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        return []

    if len(y) == 0:
        return []

    target_samples = int(TARGET_SR * TARGET_DURATION)
    spectrograms = []

    # 2. In 10s Blöcke schneiden
    for start_idx in range(0, len(y), target_samples):
        end_idx = start_idx + target_samples
        chunk = y[start_idx:end_idx]
        
        if len(chunk) == 0:
            continue

        # 3. Repeat-Padding für den letzten/kurzen Chunk
        if len(chunk) < target_samples:
            n_repeats = int(np.ceil(target_samples / len(chunk)))
            y_padded = np.tile(chunk, n_repeats)
            chunk = y_padded[:target_samples]
        
        # [cite_start]4. Mel-Spektrogramm berechnen (PaSST Specs) [cite: 143]
        n_fft = int(0.025 * TARGET_SR)      # 25ms
        hop_length = int(0.010 * TARGET_SR) # 10ms

        mel_spec = librosa.feature.melspectrogram(
            y=chunk, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=N_MELS
        )

        # 5. Logarithmieren (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        spectrograms.append(log_mel_spec)
    
    return spectrograms

def process_file(job_info):
    """Verarbeitet eine Datei und speichert das Ergebnis."""
    src_path, rel_path = job_info
    
    # Ziel-Basis-Pfad
    dst_base = OUTPUT_ROOT / rel_path.with_suffix('.npy')
    
    # Resume-Check
    check_first = dst_base.with_name(f"{dst_base.stem}_0{dst_base.suffix}")
    if check_first.exists():
        return "Skipped"

    # Berechnung starten
    specs = get_passt_spectrograms(src_path)
    
    if specs:
        # Ordner erstellen
        dst_base.parent.mkdir(parents=True, exist_ok=True)
        
        # Alle Chunks speichern
        for i, spec in enumerate(specs):
            save_path = dst_base.with_name(f"{dst_base.stem}_{i}{dst_base.suffix}")
            np.save(save_path, spec)
            
        return "Success"
    else:
        return "Error"

# --- MAIN ---
if __name__ == "__main__":
    # CPU Kerne ermitteln
    num_workers = os.cpu_count() or 1
    
    print(f"Suche MP3s in {INPUT_ROOT}...")
    all_files = list(INPUT_ROOT.rglob("*.mp3"))
    
    if not all_files:
        print("Keine Dateien gefunden!")
        exit()

    print(f"Gefunden: {len(all_files)} Dateien. Starte Verarbeitung auf {num_workers} Kernen...")
    print("(MP3-Warnungen werden unterdrückt, damit der Balken sauber bleibt)")

    # Job-Liste erstellen
    jobs = []
    for f in all_files:
        jobs.append((f, f.relative_to(INPUT_ROOT)))

    # Parallel abarbeiten
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_file, jobs), total=len(jobs), unit="file"))

    print("\nFertig! Alle Spektrogramme liegen in:", OUTPUT_ROOT)