# deepfakes-enhancement

This repository includes the scripts used in our research work. The provided inference script can be used to run the complete pipeline, which processes synthetic audio (deepfakes), enhances the phase coherence using a Vocos generator, and embeds an imperceptible spatial watermark using AudioSeal.

* The `scripts` folder contains the scripts mentioned in the thesis text.
* The `audio` folder contains several example audio files. If you want to check with other audio examples, you can access the whole dataset at: [4TTSDeepfakeData](https://huggingface.co/datasets/grotsoV/4TTSDeepfakeData).
* The checkpoint weights for the Vocos model can be downloaded at [INSERT LINK HERE]. You just need to put them in the root folder of the repo.
* The installation guide is presented in the sections below.

## Setup and Installation

1. Clone the repository and navigate into the directory:
```bash
git clone [https://github.com/Grotsov/deepfakes-enhancement.git](https://github.com/Grotsov/deepfakes-enhancement.git)
cd deepfakes-enhancement
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
> **Note for GPU Users:** To enable fast CUDA acceleration during inference, install PyTorch according to your specific system configuration from the [official PyTorch website](https://pytorch.org/get-started/locally/) *before* running the requirements installation.

## Usage

The main pipeline is contained within the inference script.

1. Open the script and modify the `[ADJUSTABLE PARAMETERS]` block to point to your specific input audio, desired output destination, and Vocos checkpoint:
```python
DEEPFAKE_PATH = "path/to/your/input.wav"
OUTPUT_PATH = "path/to/your/output.wav"
CHECKPOINT_PATH = "path/to/your/checkpoint_e4.pt"
```

2. Run the script:
```bash
python your_inference_script.py
```

The script will automatically handle sample rate conversion (to 24kHz), execute the Vocos forward pass for enhancement, apply the AudioSeal watermark, and serialize the final waveform to your specified output path.

## License

The code in this repository is licensed under the [MIT License](LICENSE).
