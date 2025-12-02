import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

def get_passt_spectrograms(
    file_path, 
    target_sr=32000, 
    target_duration=10.0, 
    n_mels=128
):
    """
    Lädt MP3, teilt sie in 10s-Blöcke, wendet Repeat-Padding an 
    und berechnet Log Mel Spektrogramme für JEDEN Block.
    Gibt eine LISTE von Spektrogrammen zurück.
    """
    try:
        # [cite_start]1. Laden & Resampling (32kHz Mono) [cite: 142]
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        return []

    if len(y) == 0:
        return []

    # Ziel-Länge in Samples (32000 * 10 = 320.000)
    target_samples = int(target_sr * target_duration)
    
    # Ergebnis-Liste
    spectrograms = []

    # 2. Schleife: Audio in 10s Blöcke schneiden
    # Wir gehen in Schritten von target_samples durch das Array
    for start_idx in range(0, len(y), target_samples):
        # Den aktuellen Schnipsel ausschneiden
        end_idx = start_idx + target_samples
        chunk = y[start_idx:end_idx]
        
        current_samples = len(chunk)
        
        # Sicherheitscheck für leere Chunks (sollte nicht passieren, aber sicher ist sicher)
        if current_samples == 0:
            continue

        # 3. Längen-Anpassung (Repeat Padding) für DIESEN Chunk
        # Wenn der Chunk kürzer als 10s ist (z.B. der letzte Rest oder kurze Dateien)
        if current_samples < target_samples:
            n_repeats = int(np.ceil(target_samples / current_samples))
            y_padded = np.tile(chunk, n_repeats)
            chunk = y_padded[:target_samples]
        
        # (Falls länger als target_samples: durch das Slicing oben [start:end] 
        # ist er automatisch exakt lang genug, also kein 'elif' nötig)

        # [cite_start]4. Mel Parameter [cite: 143]
        n_fft = int(0.025 * target_sr)      # 800 samples
        hop_length = int(0.010 * target_sr) # 320 samples

        # 5. Spektrogramm Berechnung
        mel_spec = librosa.feature.melspectrogram(
            y=chunk, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )

        # 6. Logarithmierung (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        spectrograms.append(log_mel_spec)
    
    return spectrograms

def process_file(file_info):
    """Hilfsfunktion für die Parallelisierung"""
    src_path, dst_base_path = file_info
    
    # Resume-Funktion: Wir prüfen, ob der erste Chunk (_0.npy) schon da ist.
    # Wenn ja, gehen wir davon aus, dass die Datei schon verarbeitet wurde.
    check_path = dst_base_path.with_name(f"{dst_base_path.stem}_0{dst_base_path.suffix}")
    if check_path.exists():
        return "Skipped"

    # Liste von Spektrogrammen holen
    specs = get_passt_spectrograms(src_path)
    
    if specs:
        # Zielordner erstellen
        dst_base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Alle Chunks speichern
        for i, spec in enumerate(specs):
            # Neuer Dateiname: original_0.npy, original_1.npy, etc.
            save_path = dst_base_path.with_name(f"{dst_base_path.stem}_{i}{dst_base_path.suffix}")
            np.save(save_path, spec)
            
        return "Success"
    else:
        return "Error"

# --- HAUPTPROGRAMM ---

if __name__ == "__main__":
    # Konfiguration
    INPUT_ROOT = Path("./audio_data")
    OUTPUT_ROOT = Path("./mel_spectrograms")
    
    # Automatische Erkennung der Kerne (für M1/M2/M3 Chips optimal)
    NUM_WORKERS = os.cpu_count() 

    print(f"Suche MP3-Dateien in {INPUT_ROOT}...")
    all_mp3s = list(INPUT_ROOT.rglob("*.mp3"))
    
    if not all_mp3s:
        print("Keine MP3-Dateien gefunden!")
        exit()

    print(f"{len(all_mp3s)} Dateien gefunden. Starte Verarbeitung auf {NUM_WORKERS} Kernen...")

    jobs = []
    for mp3_path in all_mp3s:
        rel_path = mp3_path.relative_to(INPUT_ROOT)
        # Wir übergeben den Basis-Pfad. Die Indizes (_0, _1) werden in process_file angehängt.
        npy_path = OUTPUT_ROOT / rel_path.with_suffix('.npy')
        jobs.append((mp3_path, npy_path))

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_file, jobs), total=len(jobs), unit="file"))

    print("\nFertig!")