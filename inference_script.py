import os
import sys
import torch
import torchaudio
from vocos import Vocos
from audioseal import AudioSeal

# --- CROSS-PLATFORM COMPATIBILITY ---
# Triton compiler that is required for Audioseal is not fully supported on Windows. This suppresses Dynamo 
# compilation errors on Windows while allowing native execution on Linux.
if sys.platform.startswith('win'):
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    print("[INFO] Windows OS detected. Triton compilation errors suppressed.")
# ------------------------------------

# --- DEVICE ALLOCATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# [ADJUSTABLE PARAMETERS]
# Modify these paths to process different audio files.
# =====================================================================
DEEPFAKE_PATH = "audio/cosy_goldengemsoflife_001_allen_0072.wav"
OUTPUT_PATH = "audio_enh/enhanced_watermarked_output.wav"
CHECKPOINT_PATH = "checkpoint_e4.pt"
# =====================================================================

def process_single_utterance(vocos_model, watermarker, deepfake_path, output_filepath):
    """
    Executes the forward pass for deepfake enhancement and applies an imperceptible
    cryptographic watermark to the resulting continuous waveform.
    """
    if not os.path.exists(deepfake_path):
        print(f"[ERROR] Source artifact not found at specified path: {deepfake_path}")
        return False
        
    try:
        print(f"[INFO] Initiating processing pipeline for: {deepfake_path}")
        wav, sr = torchaudio.load(deepfake_path)
        
        # 1. Dimensionality and Sampling Rate Standardization
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            wav = resampler(wav)
            
        wav = wav.to(DEVICE)

        # 2. Generator Inference (Phase Coherence Restoration)
        print("[INFO] Executing Vocos forward pass for spectral refinement...")
        with torch.no_grad():
            mel = vocos_model.feature_extractor(wav)
            x = vocos_model.backbone(mel)
            enhanced_wav = vocos_model.head(x)
            
            if isinstance(enhanced_wav, tuple): 
                enhanced_wav = enhanced_wav[0]
                
            enhanced_wav = torch.clamp(enhanced_wav, min=-0.99, max=0.99)
            
            if enhanced_wav.dim() == 3:
                enhanced_wav = enhanced_wav.squeeze(1)

            # 3. Cryptographic Watermarking (AudioSeal)
            print("[INFO] Embedding AudioSeal spatial watermark into the refined tensor...")
            watermark = watermarker.get_watermark(enhanced_wav.unsqueeze(0))
            final_audio = enhanced_wav + watermark.squeeze(0)

        # 4. Serialization
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        torchaudio.save(output_filepath, final_audio.cpu(), 24000)
        
        print(f"[SUCCESS] Enhanced and watermarked artifact serialized to: {output_filepath}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Exception encountered during forward pass: {e}")
        return False

def main():
    print(f"[INFO] Allocating generator architecture to {DEVICE}...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint state dictionary not located at: {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'vocos_state_dict' in checkpoint:
        vocos.load_state_dict(checkpoint['vocos_state_dict'])
    else:
        vocos.load_state_dict(checkpoint)
    vocos.eval()

    print(f"[INFO] Instantiating AudioSeal localization model on {DEVICE}...")
    watermarker = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
    
    process_single_utterance(vocos, watermarker, DEEPFAKE_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    main()