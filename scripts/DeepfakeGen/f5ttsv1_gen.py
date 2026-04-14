"""
F5-TTS Single-Shot Deepfake Generation Pipeline
--------------------------------------------------
Description: A multi-processing inference script for the F5-TTS (DiT) architecture.
             Optimized for single-GPU execution with high VRAM availability.
Dependencies: torch, f5_tts, ffmpeg, soundfile
"""

import sys
import json
import os
import time
import queue
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

# 1. Strict CPU thread limitation and VRAM fragmentation management
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CONFIGURATION ---
MANIFEST_PATH = "./data/master_manifest_final.json" 
BASE_AUDIO_DIR = "./data/recordings/" 
OUTPUT_DIR = "./output/f5tts_full/"
CKPT_PATH = "./models/checkpoints/F5TTS_v1_Base/model_1250000.safetensors"

# Configuration for high-memory GPUs (e.g., RTX 6000 48GB)
NUM_WORKERS = 3  

os.makedirs(OUTPUT_DIR, exist_ok=True)

def worker_process(task_queue, output_queue, worker_id):
    # Suppress verbose output for stability
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import logging
    logging.getLogger().setLevel(logging.CRITICAL) 
    f = open(os.devnull, 'w')
    sys.stdout = f
    sys.stderr = f

    # Lazy imports to ensure proper CUDA initialization in sub-processes
    import torch
    from f5_tts.infer.utils_infer import load_model, infer_process, load_vocoder
    from f5_tts.model import DiT

    # Staggered start to avoid VRAM allocation spikes
    time.sleep(worker_id * 7)

    device = "cuda:0"
    torch.cuda.set_device(0)
    
    try:
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        model_obj = load_model(
            model_cls=DiT,
            model_cfg=model_cfg,
            ckpt_path=CKPT_PATH,
            mel_spec_type="vocos",
            vocab_file="",
            ode_method="euler",
            use_ema=True,
            device=device
        )
        vocoder = load_vocoder()
    except Exception as e:
        return 

    while True:
        try:
            item = task_queue.get(timeout=10)
        except queue.Empty:
            break

        if item == "STOP":
            break
            
        raw_path = item['audio_filepath'].replace("\\", "/")
        rel_path = raw_path.split('recordings/')[-1] if 'recordings/' in raw_path else Path(raw_path).name
        
        prompt_wav = os.path.join(BASE_AUDIO_DIR, rel_path)
        output_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        prompt_text = item['text']
        target_text = item['text']
        
        result_data = {
            "model_name": "f5tts",
            "original_filename": Path(raw_path).name,
            "generated_filepath": output_path,
            "status": "failed",
            "gpu_used": "cuda:0"
        }

        # Check for existing files to support process resumption
        if os.path.exists(output_path):
            result_data["status"] = "success"
            output_queue.put(result_data)
            continue
        
        try:
            # Inference process with 32 NFE steps
            wav, sr, _ = infer_process(
                prompt_wav,
                prompt_text,
                target_text,
                model_obj,
                vocoder,
                device=device,
                nfe_step=32
            )
            
            sf.write(output_path, wav, sr)
            result_data["status"] = "success"
            
        except Exception as e:
            result_data["error_message"] = str(e)
        finally:
            output_queue.put(result_data)
            torch.cuda.empty_cache()

def main():
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    total_tasks = len(all_tasks)
    print(f"Tasks loaded: {total_tasks}. Utilizing {NUM_WORKERS} workers on GPU 0.")
    
    task_queue = mp.Queue()
    output_queue = mp.Queue()
    processes = []
    
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(task_queue, output_queue, i))
        p.start()
        processes.append(p)

    start_total = time.time()
    
    for task in all_tasks:
        task_queue.put(task)
        
    for _ in range(NUM_WORKERS):
        task_queue.put("STOP")

    log_file_path = os.path.join(OUTPUT_DIR, "f5tts_generation_log.jsonl")
    successful_count = 0

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        with tqdm(total=total_tasks, desc="F5-TTS Generation") as pbar:
            completed_tasks = 0
            while completed_tasks < total_tasks:
                try:
                    result = output_queue.get(timeout=600)
                    if result["status"] == "success":
                        log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        log_file.flush()
                        successful_count += 1
                    completed_tasks += 1
                    pbar.update(1)
                except queue.Empty:
                    break

    for p in processes:
        p.terminate()
        p.join()

    total_time = time.time() - start_total
    print(f"\nExecution Time: {total_time:.2f} seconds.")
    print(f"Log saved to: {log_file_path} (Successful: {successful_count}/{total_tasks})")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()