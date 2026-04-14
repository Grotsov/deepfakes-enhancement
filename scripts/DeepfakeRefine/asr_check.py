import json
import os
import argparse
import numpy as np
from faster_whisper import WhisperModel
from whisper.normalizers import EnglishTextNormalizer
import jiwer
from tqdm import tqdm
import concurrent.futures

# ================= PRODUCTION THRESHOLDS =================
WER_THRESHOLD = 0.07      
HEAD_WORDS_COUNT = 3      
LOGPROB_THRESHOLD = -0.95 

# Initialize model and normalizer
model = WhisperModel(
    "large-v3", 
    device="cuda", 
    compute_type="float16",
    num_workers=12
)
whisper_norm = EnglishTextNormalizer()

def normalize_text(text):
    if not text: return ""
    return whisper_norm(text)

def process_line(line, reference_texts, base_audio_dir):
    try:
        entry = json.loads(line)
        gen_rel_path = entry.get("generated_filepath") or entry.get("generated_web_path")
        if not gen_rel_path: return None
        
        # Ensure absolute pathing
        full_audio_path = os.path.join(base_audio_dir, gen_rel_path.replace("\\", "/"))
        orig_raw = entry.get("original_filename") or entry.get("original_filepath") or ""
        orig_fname = os.path.basename(orig_raw.replace("\\", "/"))
        
        expected_text = reference_texts.get(orig_fname)
        
        # Safeguard against missing files
        if not os.path.exists(full_audio_path): return None
        if not expected_text: return None

        # Transcription execution
        segments_gen, _ = model.transcribe(full_audio_path, beam_size=5, vad_filter=False)
        segments = list(segments_gen)
        transcribed_text = " ".join([s.text for s in segments])
        
        # Calculate average acoustic confidence
        avg_logprob = np.mean([s.avg_logprob for s in segments]) if segments else -2.0
        
        # NORMALIZE BOTH TEXTS
        norm_expected = normalize_text(expected_text)
        norm_transcribed = normalize_text(transcribed_text)
        
        ref_words = norm_expected.split()
        hyp_words = norm_transcribed.split()
        
        if not ref_words or not hyp_words: return None
            
        # --- VERIFICATION LOGIC ---
        head_len = min(HEAD_WORDS_COUNT, len(ref_words), len(hyp_words))
        is_bad_head = ref_words[:head_len] != hyp_words[:head_len]
        
        wer_score = jiwer.wer(norm_expected, norm_transcribed)
        is_bad_wer = wer_score > WER_THRESHOLD
        
        # Compound Word Heuristic (e.g., warhorse vs war horse)
        if is_bad_wer or is_bad_head:
            if norm_expected.replace(" ", "") == norm_transcribed.replace(" ", ""):
                is_bad_wer = False
                is_bad_head = False
                wer_score = 0.0
        
        is_bad_conf = avg_logprob < LOGPROB_THRESHOLD

        # If flagged as corrupted
        if is_bad_head or is_bad_wer or is_bad_conf:
            model_name = entry.get("model_name", "unknown")
            reasons = []
            if is_bad_head: reasons.append(f"HEAD_FAIL")
            if is_bad_wer: reasons.append(f"WER_{wer_score:.2f}")
            if is_bad_conf: reasons.append(f"CONF_{avg_logprob:.2f}")
            
            return {
                "model": model_name,
                "file": orig_fname,
                "reasons": reasons,
                "expected_raw": expected_text,
                "transcribed_raw": transcribed_text,
                "wer_clean": round(wer_score, 3)
            }
            
    except Exception: return None
    return None

def main(args):
    # 1. Load ground truth reference texts
    print(f"Loading original reference manifest: {args.master_manifest}...")
    reference_texts = {}
    with open(args.master_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                if "audio_filepath" in entry:
                    fname = os.path.basename(entry["audio_filepath"])
                    reference_texts[fname] = entry["text"]
            except: continue

    # 2. Execution Engine
    print(f"Loaded ground truth references: {len(reference_texts)}. Commencing audit for {args.model_name}...")
    
    with open(args.generated_manifest, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Pass the arguments into the worker threads using a lambda or wrapper
    process_wrapper = lambda line: process_line(line, reference_texts, args.base_audio_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_wrapper, lines), total=len(lines), desc=f"Auditing {args.model_name}"))

    mismatched_data = [r for r in results if isinstance(r, dict)]

    # 3. Save results to manifest
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(mismatched_data, f, indent=4, ensure_ascii=False)

    print(f"\nAudit Complete for {args.model_name}. Total failures identified: {len(mismatched_data)} out of {len(lines)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized Whisper ASR Evaluation Pipeline")
    parser.add_argument("--model_name", required=True, help="Name of the TTS model being evaluated")
    parser.add_argument("--master_manifest", default="master_manifest_final.json", help="Path to ground truth manifest")
    parser.add_argument("--generated_manifest", required=True, help="Path to the TTS generation log")
    parser.add_argument("--base_audio_dir", required=True, help="Root directory of the generated audio files")
    parser.add_argument("--output_file", required=True, help="Output JSON file for mismatches")
    parser.add_argument("--workers", type=int, default=40, help="Number of concurrent worker threads")
    
    arguments = parser.parse_args()
    main(arguments)