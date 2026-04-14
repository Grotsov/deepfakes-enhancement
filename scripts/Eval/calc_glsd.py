import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

def calc_global_lsd(ref_path: str, deg_path: str) -> float:
    """
    Calculates the Log-Spectral Distance (LSD) between a reference 
    and a degraded audio signal using STFT.
    """
    sr_ref, ref = wavfile.read(ref_path)
    sr_deg, deg = wavfile.read(deg_path)
    
    # 1. Amplitude Normalization
    if ref.dtype == np.int16: ref = ref.astype(np.float32) / 32768.0
    if deg.dtype == np.int16: deg = deg.astype(np.float32) / 32768.0
    
    # 2. Mono Conversion
    if len(ref.shape) > 1: ref = ref.mean(axis=1)
    if len(deg.shape) > 1: deg = deg.mean(axis=1)
    
    # 3. STFT Parameters (Window size: 2048)
    _, _, Z_ref = stft(ref, fs=sr_ref, nperseg=2048)
    _, _, Z_deg = stft(deg, fs=sr_deg, nperseg=2048)
    
    # 4. Logarithmic Power Spectrum Extraction
    s_ref = 10 * np.log10(np.mean(np.abs(Z_ref)**2, axis=1) + 1e-8)
    s_deg = 10 * np.log10(np.mean(np.abs(Z_deg)**2, axis=1) + 1e-8)
    
    # 5. Mean Squared Error of the Spectra
    lsd_value = np.sqrt(np.mean((s_ref - s_deg)**2))
    
    return float(lsd_value)
