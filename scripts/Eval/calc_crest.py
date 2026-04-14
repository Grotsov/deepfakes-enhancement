import numpy as np
import librosa

def calculate_crest_factor(path: str) -> float:
    """
    Computes the Crest Factor (Peak-to-RMS ratio) to quantify 
    non-stationary transients and dynamic energy distribution.
    """
    # Load audio preserving native sample rate
    y, _ = librosa.load(path, sr=None)
    
    if len(y) == 0: 
        return None
        
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    
    if rms > 0:
        return float(peak / rms)
    return None