import os
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= SETTINGS =================
TARGET_LUFS = -20.0
TRUE_PEAK_LIMIT_DB = -1.0
LINEAR_PEAK_LIMIT = 10 ** (TRUE_PEAK_LIMIT_DB / 20)

DIRECTORIES = [
    "recordings",
    "mask_gct",
    "fish_speech",
    "f5tts",
    "cosy_voice"
]

def process_audio(file_path):
    try:
        data, rate = sf.read(file_path)
        
        # Measure BS.1770 integrated loudness
        meter = pyln.Meter(rate)
        current_loudness = meter.integrated_loudness(data)
        
        # Normalize
        normalized_data = pyln.normalize.loudness(data, current_loudness, TARGET_LUFS)
        
        # Peak Limit Verification
        peak = np.max(np.abs(normalized_data))
        was_clipping = False
        
        if peak > LINEAR_PEAK_LIMIT:
            was_clipping = True
            # Apply hard limiting to prevent 16-bit distortion
            normalized_data = normalized_data * (LINEAR_PEAK_LIMIT / peak)
            
        sf.write(file_path, normalized_data, rate, subtype='PCM_16')
        return True, was_clipping
        
    except Exception as e:
        return False, str(e)

def main():
    print(f"Initiating Normalization: {TARGET_LUFS} LUFS, Peak Limit: {TRUE_PEAK_LIMIT_DB} dB")
    clipping_files = []

    for folder in DIRECTORIES:
        folder_path = Path(folder)
        if not folder_path.exists(): continue

        print(f"\nProcessing: {folder}")
        wav_files = list(folder_path.rglob("*.wav"))
        success, errors = 0, 0

        for wav_path in tqdm(wav_files, desc="Processing"):
            ok, result = process_audio(wav_path)
            if ok:
                success += 1
                if result is True: clipping_files.append(str(wav_path))
            else: errors += 1

        print(f"Completed {folder}: {success} successful, {errors} failed.")

    print("\n" + "="*50)
    if clipping_files:
        print(f"WARNING: Hard limiting applied to {len(clipping_files)} files to prevent clipping.")
    else:
        print("Success: Zero instances of digital clipping detected.")
    print("="*50)

if __name__ == "__main__":
    main()