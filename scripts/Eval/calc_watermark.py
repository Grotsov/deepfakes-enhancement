import torch
from audioseal import AudioSeal
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_eer(y_true: list, y_score: list) -> float:
    """
    Calculates the Equal Error Rate (EER) based on detection scores.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return float(eer)
    except Exception:
        return 0.0

def apply_and_detect_watermark(wav_tensor: torch.Tensor, device: str = "cuda"):
    """
    Core logic for 16-bit payload injection, detection, and bit-accuracy scoring.
    """
    generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    
    # 1. Generate random 16-bit message payload
    msg = torch.randint(0, 2, (1, 16), device=device) 
    
    # 2. Inject Watermark
    with torch.no_grad():
        watermark = generator.get_watermark(wav_tensor, sample_rate=16000, message=msg)
    watermarked_wav = wav_tensor + watermark
    
    # 3. Detect Watermark (Positive Class)
    with torch.no_grad():
        res_wm, detected_msg = detector.detect_watermark(watermarked_wav, sample_rate=16000)
    
    # Calculate Bit Accuracy
    bit_accuracy = (detected_msg == msg).float().mean().item()
    detection_score = res_wm.item()
    
    return watermarked_wav, bit_accuracy, detection_score