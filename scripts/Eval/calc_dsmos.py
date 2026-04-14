import numpy as np
import librosa
import onnxruntime as ort

class DNSEvaluator:
    def __init__(self, model_path: str = "sig_bak_ovr.onnx"):
        # Initialize ONNX session with CUDA execution provider
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            }),
            'CPUExecutionProvider'
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_score(self, audio_path: str) -> float:
        """
        Extracts the Overall Quality (OVR) DNSMOS score.
        """
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # Target length matching the DNSMOS training constraints
        target_len = 144160
        audio = np.pad(audio, (0, max(0, target_len - len(audio))))[:target_len]
        
        input_data = audio.reshape(1, -1).astype(np.float32)
        res = self.session.run(None, {self.input_name: input_data})[0]
        
        # Return the OVR (Overall) metric score
        return float(res[0][2])