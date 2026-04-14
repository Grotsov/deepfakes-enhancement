"""
MaskGCT Single-Shot Deepfake Generation Pipeline
--------------------------------------------------
Description: A multi-processing inference script designed to generate 
             non-parallel synthetic speech datasets using the MaskGCT architecture. 
             This script enforces a 1:1 single-shot generation protocol by utilizing 
             the bona fide audio as both the acoustic prompt and text reference.
Dependencies: espeak-ng, ffmpeg, torch, safetensors, soundfile
"""

import json
import os
import time
import torch
import queue
import multiprocessing as mp
from pathlib import Path
import soundfile as sf
import safetensors.torch
from tqdm import tqdm

# Strict CPU thread limitation to prevent bottlenecks during parallel tensor operations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from models.tts.maskgct.maskgct_utils import *

# Weight loading patch for safetensors compatibility
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# --- RELATIVE PATH CONFIGURATION ---
MANIFEST_PATH = "./data/master_manifest_final.json"
BASE_AUDIO_DIR = "./data/recordings/" 
OUTPUT_DIR = "./output/maskgct_full/"
CKPT_DIR = "./models/checkpoints/"

# --- MULTI-GPU SETTINGS ---
NUM_GPUS = 3
WORKERS_PER_GPU = 3 
NUM_WORKERS = NUM_GPUS * WORKERS_PER_GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

def worker_process(task_queue, output_queue, worker_id, num_gpus):
    # Staggered initialization to prevent OOM errors during simultaneous model loading
    time.sleep(worker_id * 6)
    
    gpu_id = worker_id % num_gpus
    device = torch.device(f"cuda:{gpu_id}")
    
    cfg = load_config("./models/tts/maskgct/config/maskgct.json")

    # Module initialization
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # Safetensor weight loading
    safetensors.torch.load_model(semantic_codec, os.path.join(CKPT_DIR, "semantic_codec/model.safetensors"))
    safetensors.torch.load_model(codec_encoder, os.path.join(CKPT_DIR, "acoustic_codec/model.safetensors"))
    safetensors.torch.load_model(codec_decoder, os.path.join(CKPT_DIR, "acoustic_codec/model_1.safetensors"))
    safetensors.torch.load_model(t2s_model, os.path.join(CKPT_DIR, "t2s_model/model.safetensors"))
    safetensors.torch.load_model(s2a_model_1layer, os.path.join(CKPT_DIR, "s2a_model/s2a_model_1layer/model.safetensors"))
    safetensors.torch.load_model(s2a_model_full, os.path.join(CKPT_DIR, "s2a_model/s2a_model_full/model.safetensors"))

    maskgct_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model, semantic_codec, codec_encoder, codec_decoder, 
        t2s_model, s2a_model_1layer, s2a_model_full, 
        semantic_mean, semantic_std, device
    )

    while True:
        try:
            item = task_queue.get(timeout=10)
        except queue.Empty: break
        if item == "STOP": break
            
        raw_path = item['audio_filepath'].replace("\\", "/")
        rel_path = raw_path.split('recordings/')[-1] if 'recordings/' in raw_path else Path(raw_path).name
        
        prompt_wav = os.path.join(BASE_AUDIO_DIR, rel_path)
        
        # Hierarchical output directory mirroring
        output_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result_data = {
            "model_name": "maskgct",
            "original_filename": Path(raw_path).name,
            "generated_filepath": output_path,
            "status": "failed",
            "gpu_used": f"cuda:{gpu_id}"
        }

        # Fault tolerance and skip logic
        if os.path.exists(output_path):
            result_data["status"] = "success"
            output_queue.put(result_data)
            continue

        if not os.path.exists(prompt_wav):
            result_data["error_message"] = "Source not found"
            output_queue.put(result_data)
            continue
        
        try:
            # Single-shot execution: Identical text for both prompt and target
            audio = maskgct_pipeline.maskgct_inference(prompt_wav, item['text'], item['text'], language="en")
            sf.write(output_path, audio, 24000)
            result_data["status"] = "success"
        except Exception as e:
            result_data["error_message"] = str(e)
        finally:
            output_queue.put(result_data)
            torch.cuda.empty_cache() 

def main():
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        all_tasks = [json.loads(line) for line in f]

    print(f"Tasks loaded: {len(all_tasks)}")
    task_queue = mp.Queue()
    output_queue = mp.Queue()
    processes = [mp.Process(target=worker_process, args=(task_queue, output_queue, i, NUM_GPUS)) for i in range(NUM_WORKERS)]
    
    for p in processes: p.start()
    for t in all_tasks: task_queue.put(t)
    for _ in range(NUM_WORKERS): task_queue.put("STOP")

    # Asynchronous logging execution
    log_file_path = os.path.join(OUTPUT_DIR, "maskgct_generation_log.jsonl")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        for _ in tqdm(range(len(all_tasks)), desc="Generation Progress"):
            result = output_queue.get()
            log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            log_file.flush()

    for p in processes: p.join()
    print(f"Generation complete. Log saved to: {log_file_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()