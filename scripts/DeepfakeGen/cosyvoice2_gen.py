"""
CosyVoice 2 Parallelized Generation Pipeline
-------------------------------------------
Description: A high-concurrency multi-GPU inference script utilizing 
             deferred initialization and hardware isolation.
Dependencies: torch, torchaudio, cosyvoice
"""

import sys
import json
import os
import time
import queue
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# Environmental constraints for threading stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- CONFIGURATION ---
MANIFEST_PATH = "./data/master_manifest_final.json"
BASE_AUDIO_DIR = "./data/recordings/" 
OUTPUT_DIR = "./output/cosyvoice2_full"
MODEL_PATH = './models/pretrained_models/CosyVoice2-0.5B'

NUM_WORKERS = 6 
GPUS = [0, 1] # Dual-GPU deployment

os.makedirs(OUTPUT_DIR, exist_ok=True)

def worker_process(task_queue, output_queue, worker_id):
    """
    Worker lifecycle:
    1. Isolate physical GPU.
    2. Defer PyTorch/Model imports until process is spawned.
    3. Execute zero-shot inference.
    """
    # Hardware Isolation
    physical_gpu_id = GPUS[worker_id % len(GPUS)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    # Deferred Imports to prevent CUDA context collisions
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2

    # Suppress console verbosity for clean logging
    import logging
    logging.getLogger().setLevel(logging.CRITICAL) 
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull

    # Context initialization
    device = "cuda:0"
    torch.cuda.set_device(0)
    
    try:
        model = CosyVoice2(MODEL_PATH)
    except Exception as e:
        return

    while True:
        item = task_queue.get()
        if item == "STOP": 
            break
            
        raw_path = item['audio_filepath'].replace("\\", "/")
        rel_path = raw_path.split('recordings/')[-1] if 'recordings/' in raw_path else Path(raw_path).name
        
        prompt_wav = os.path.join(BASE_AUDIO_DIR, rel_path)
        output_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result_data = {
            "model_name": "cosyvoice2",
            "original_filename": Path(raw_path).name,
            "generated_filepath": output_path,
            "status": "failed",
            "gpu_id": physical_gpu_id
        }

        if os.path.exists(output_path):
            result_data["status"] = "success"
            output_queue.put(result_data)
            continue

        try:
            # Zero-shot inference execution
            generator = model.inference_zero_shot(
                item['text'], 
                item['text'], 
                prompt_wav, 
                stream=False
            )
            
            # Concatenate phonetic chunks and save at 24kHz
            audio_data = torch.cat([i['tts_speech'].cpu() for i in generator], dim=1)
            torchaudio.save(output_path, audio_data, 24000)
            
            result_data["status"] = "success"
        except Exception as e:
            result_data["error_message"] = str(e)
        finally:
            output_queue.put(result_data)
            torch.cuda.empty_cache()

def main():
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    task_queue = mp.Queue()
    output_queue = mp.Queue()
    
    processes = [mp.Process(target=worker_process, args=(task_queue, output_queue, i)) for i in range(NUM_WORKERS)]
    
    for p in processes: 
        p.start()
    
    for t in all_tasks: 
        task_queue.put(t)
    
    for _ in range(NUM_WORKERS): 
        task_queue.put("STOP")

    log_file_path = os.path.join(OUTPUT_DIR, "cosy2_generation_log.jsonl")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        for _ in tqdm(range(len(all_tasks)), desc="CosyVoice2 Generation"):
            res = output_queue.get()
            log_file.write(json.dumps(res, ensure_ascii=False) + "\n")
            log_file.flush()

    for p in processes: 
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()