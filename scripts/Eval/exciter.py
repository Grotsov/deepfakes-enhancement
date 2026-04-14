import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from tqdm import tqdm
import argparse

def dsp_harmonic_exciter(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Applies a classical DSP Harmonic Bandwidth Extension (Exciter).
    
    This algorithm serves as a traditional, deterministic baseline for 
    high-frequency reconstruction. It utilizes non-linear distortion 
    (soft-clipping) to synthesize artificial upper harmonics from the 
    mid-range signal.
    
    Algorithm steps strictly follow the thesis methodology (Section 7.7):
    1. 4th-order Butterworth high-pass filter at 3 kHz.
    2. Symmetric soft-clipping (hyperbolic tangent) with a drive multiplier of 5.0.
    3. 4 kHz high-pass filter to clean synthetic harmonics.
    4. Blend back into the original waveform at a 15% mix ratio.
    5. Strict peak normalization to prevent digital clipping.
    """
    nyq = 0.5 * sr
    
    # Step 1: Isolate upper-midband frequencies (High-pass at 3 kHz)
    b_mid, a_mid = signal.butter(4, 3000 / nyq, btype='high')
    high_freq_band = signal.lfilter(b_mid, a_mid, audio)
    
    # Step 2: Non-linear distortion to synthesize harmonics
    drive = 5.0
    harmonics = np.tanh(high_freq_band * drive)
    
    # Step 3: Clean the synthesized harmonics (High-pass at 4 kHz)
    b_high, a_high = signal.butter(4, 4000 / nyq, btype='high')
    harmonics_clean = signal.lfilter(b_high, a_high, harmonics)
    
    # Step 4: Blend with the original signal (15% mix ratio)
    mix_amount = 0.15
    enhanced_audio = audio + (harmonics_clean * mix_amount)
    
    # Step 5: Strict peak normalization (ceiling at 0.95 to prevent clipping)
    max_peak = np.max(np.abs(enhanced_audio))
    if max_peak > 1.0:
        enhanced_audio = (enhanced_audio / max_peak) * 0.95
        
    return enhanced_audio

def process_dataset(input_base_dir: str, sr: int = 16000):
    """
    Iterates through the generated deepfake directories and applies 
    the traditional DSP baseline to each audio file.
    """
    models = ["cosy_voice", "f5tts", "fish_speech", "mask_gct"]
    
    print("\nGENERATING TRADITIONAL DSP BASELINE (Harmonic Exciter)")
    print("=" * 65)
    
    for model in models:
        dir_in = os.path.join(input_base_dir, f"fad_{model}_dirty")
        dir_out = os.path.join(input_base_dir, f"fad_{model}_traditional_enhanced")
        
        if not os.path.exists(dir_in):
            print(f"Directory not found, skipping: {dir_in}")
            continue
            
        os.makedirs(dir_out, exist_ok=True)
        files = [f for f in os.listdir(dir_in) if f.endswith('.wav')]
        
        for f in tqdm(files, desc=f"Processing {model}", leave=False):
            try:
                path_in = os.path.join(dir_in, f)
                path_out = os.path.join(dir_out, f)
                
                # Load audio
                wav, _ = librosa.load(path_in, sr=sr)
                
                # Apply traditional exciter
                wav_trad = dsp_harmonic_exciter(wav, sr)
                
                # Save processed audio
                sf.write(path_out, wav_trad, sr)
                
            except Exception as e:
                print(f"Error processing {f}: {e}")

    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply DSP Harmonic Exciter as a baseline for deepfake enhancement.")
    parser.add_argument("--data_dir", type=str, default=".", help="Base directory containing the FAD audio folders.")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    args = parser.parse_args()
    
    process_dataset(args.data_dir, args.sr)
