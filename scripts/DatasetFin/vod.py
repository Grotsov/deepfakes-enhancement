import os
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= SETTINGS =================
TOP_DB = 40.0         
MARGIN_MS = 50        
DIRECTORIES = [
    "recordings",
    "mask_gct",
    "fish_speech",
    "f5tts",
    "cosy_voice"
]

def main():
    print(f"Initiating VAD Trimming: Threshold -{TOP_DB}dB, Margin {MARGIN_MS}ms")
    
    total_trimmed = 0
    total_seconds_saved = 0.0

    for folder in DIRECTORIES:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Directory not found: {folder}")
            continue

        print(f"\nProcessing directory: {folder}")
        wav_files = list(folder_path.rglob("*.wav"))
        
        for wav_path in tqdm(wav_files, desc="Trimming files"):
            try:
                y, sr = sf.read(wav_path)
                
                # Force mono array for Librosa compatibility
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                
                initial_duration = len(y) / sr
                
                # Detect signal boundaries
                _, index = librosa.effects.trim(y, top_db=TOP_DB)
                
                # Apply safety margin to prevent fricative clipping
                margin_samples = int(sr * (MARGIN_MS / 1000.0))
                start_safe = max(0, index[0] - margin_samples)
                end_safe = min(len(y), index[1] + margin_samples)
                
                trimmed_y = y[start_safe:end_safe]
                
                # Overwrite only if silence was successfully removed
                if len(trimmed_y) < len(y):
                    sf.write(wav_path, trimmed_y, sr, subtype='PCM_16')
                    total_trimmed += 1
                    total_seconds_saved += (initial_duration - (len(trimmed_y) / sr))
                    
            except Exception as e:
                print(f"Error processing file {wav_path}: {e}")

    print("\n" + "="*50)
    print(f"Process Complete! Total files trimmed: {total_trimmed}")
    print(f"Total digital silence removed: {total_seconds_saved:.2f} seconds.")
    print("="*50)

if __name__ == "__main__":
    main()