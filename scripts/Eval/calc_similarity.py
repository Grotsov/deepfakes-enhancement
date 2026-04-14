import torch
import numpy as np
import librosa
from speechbrain.inference.classifiers import EncoderClassifier

class SpeakerVerifier:
    def __init__(self, device: str = "cuda"):
        # Load pre-trained ECAPA-TDNN VoxCeleb model
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )

    def get_embedding(self, audio_path: str) -> np.ndarray:
        """Extracts the 192-dimensional speaker embedding vector."""
        signal, fs = librosa.load(audio_path, sr=16000)
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.classifier.encode_batch(signal_tensor)
            return emb.squeeze().cpu().numpy()

def cos_sim(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Calculates the cosine similarity between two embedding vectors."""
    return np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))