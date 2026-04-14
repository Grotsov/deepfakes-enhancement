"""
Fish-S1mini Distributed Generation Pipeline
-------------------------------------------
Description: A multi-threaded inference script utilizing a load-balanced 
             API architecture for high-throughput speech synthesis.
Dependencies: requests, torchaudio, tqdm, concurrent.futures
"""

import os
import json
import base64
import time
import requests
import io
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
MANIFEST_PATH = "./data/master_manifest_final.json" 
BASE_AUDIO_DIR = "./data/recordings/"
OUTPUT_DIR = "./output/fish_s1_full"

# Distributed API Endpoints
API_URLS = [
    "http://127.0.0.1:8080", 
    "http://127.0.0.1:8082", 
    "http://127.0.0.1:8083"
]

TARGET_SR = 24000 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_task(args):
    """
    Processes a single synthesis task via a load-balanced API request.
    Includes automated resampling to TARGET_SR.
    """
    item, global_idx = args
    
    # Load Balancer: Distribute tasks across available API ports
    api_url = f"{API_URLS[global_idx % len(API_URLS)]}/v1/tts"
    
    raw_path = item['audio_filepath'].replace("\\", "/")
    rel_path = raw_path.split('recordings/')[-1] if 'recordings/' in raw_path else Path(raw_path).name
    
    prompt_wav_path = os.path.join(BASE_AUDIO_DIR, rel_path)
    output_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    target_text = item['text']
    prompt_text = item['text']
    
    result_data = {
        "model_name": "fish_s1",
        "original_filename": Path(raw_path).name,
        "generated_filepath": output_path,
        "status": "failed",
        "api_endpoint": api_url
    }

    if os.path.exists(output_path):
        result_data["status"] = "success"
        return result_data

    try:
        with open(prompt_wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            
        payload = {
            "text": target_text,
            "references": [{"audio": audio_b64, "text": prompt_text}],
            "format": "wav"
        }
        
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            audio_bytes = io.BytesIO(response.content)
            waveform, sr_orig = torchaudio.load(audio_bytes)
            
            # Automated resampling to standardized frequency
            if sr_orig != TARGET_SR:
                resampler = T.Resample(sr_orig, TARGET_SR)
                waveform = resampler(waveform)
            
            torchaudio.save(output_path, waveform, TARGET_SR)
            result_data["status"] = "success"
        else:
            result_data["error_message"] = f"API Error {response.status_code}"
            
    except Exception as e:
        result_data["error_message"] = str(e)
        
    return result_data

def main():
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    total_tasks = len(all_tasks)
    work_queue = [(task, idx) for idx, task in enumerate(all_tasks)]
    
    log_file_path = os.path.join(OUTPUT_DIR, "fish_s1_generation_log.jsonl")
    successful_count = 0
    start_time = time.time()

    # Worker count optimized for network-GPU concurrency
    max_workers = len(API_URLS) * 2 
    
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(process_task, work_queue), total=total_tasks):
                if result["status"] == "success":
                    # Strip API internal data for clean logging
                    log_entry = {k: v for k, v in result.items() if k != "api_endpoint"}
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    log_file.flush()
                    successful_count += 1

    total_time = time.time() - start_time
    print(f"\nExecution Complete. Successfully generated: {successful_count}/{total_tasks}")
    print(f"Total Time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()