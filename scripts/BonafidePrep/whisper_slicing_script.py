import json
import os
import torch
import torchaudio
import whisperx
import re
from tqdm import tqdm

# Configuration Parameters
INPUT_MANIFEST = "dataset_long_files.jsonl"
OUTPUT_DIR = "dataset_sliced"
FINAL_MANIFEST = "dataset_sliced.jsonl"

MAX_DUR = 9.8         # Absolute maximum duration threshold
MIN_LOOK = 3.0        # Minimum duration before evaluating pause boundaries
MIN_PAUSE_LEN = 0.4   # Minimum inter-word silence required for natural split
BUFFER_SEC = 0.05     # Safety buffer around acoustic boundaries
TAIL_PAD_SEC = 0.1    # Trailing silence padding
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def align_punctuation(original_text, whisper_words):
    """Reconstructs source punctuation by aligning normalized word segments."""
    orig_tokens = original_text.split()
    clean_orig = [re.sub(r'[^\w\s-]', '', t).lower() for t in orig_tokens]
    aligned_text_parts = []
    orig_ptr = 0
    
    for w_obj in whisper_words:
        w_text = re.sub(r'[^\w\s-]', '', w_obj['word']).lower()
        found = False
        for j in range(orig_ptr, min(orig_ptr + 40, len(clean_orig))):
            if w_text == clean_orig[j]:
                aligned_text_parts.append(orig_tokens[j])
                orig_ptr = j + 1
                found = True
                break
        if not found: 
            aligned_text_parts.append(w_obj['word'])
            
    return " ".join(aligned_text_parts)

def save_safe_chunk(waveform, text, out_path, sr, entries, is_forced=False):
    """Exports audio with adaptive fade-out and silence padding."""
    fade_duration = 0.07 if is_forced else 0.03
    fade_len = int(fade_duration * sr)
    
    if waveform.shape[1] > fade_len:
        fade_curve = torch.linspace(1.0, 0.0, fade_len)
        waveform[:, -fade_len:] *= fade_curve

    silence_tail = torch.zeros((waveform.shape[0], int(TAIL_PAD_SEC * sr)))
    final_wave = torch.cat([waveform, silence_tail], dim=1)

    torchaudio.save(out_path, final_wave, sr)
    
    entries.append({
        "audio_filepath": os.path.abspath(out_path),
        "duration": final_wave.shape[1] / sr,
        "text": text
    })

def run_slicing_pipeline():
    """Executes the forced-cut protected acoustic slicing pipeline."""
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    final_entries = []

    with open(INPUT_MANIFEST, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc="Processing Segments"):
        data = json.loads(line)
        audio_path = data["audio_filepath"]
        full_text = data.get("text_normalized", data["text"])
        
        try:
            audio = whisperx.load_audio(audio_path)
            waveform, sr = torchaudio.load(audio_path)
            res = whisperx.align(
                [{"text": full_text, "start": 0, "end": waveform.shape[1]/sr}], 
                model_a, metadata, audio, DEVICE
            )
            words = [w for w in res["word_segments"] if 'start' in w]
            
            if not words: continue

            curr_idx, split_idx = 0, 0
            total_samples = waveform.shape[1]

            while curr_idx < len(words):
                start_sec = words[curr_idx]['start']
                best_pause_val = -1
                last_word_idx = backup_last_idx = curr_idx
                
                for i in range(curr_idx, len(words)):
                    w = words[i]
                    dur = w['end'] - start_sec
                    
                    if dur > MAX_DUR: break
                    backup_last_idx = i
                    
                    if i < len(words) - 1:
                        pause_len = words[i+1]['start'] - w['end']
                        if dur >= MIN_LOOK and pause_len >= MIN_PAUSE_LEN:
                            if pause_len > best_pause_val:
                                best_pause_val = pause_len
                                last_word_idx = i

                is_forced_cut = (best_pause_val == -1)
                if is_forced_cut: last_word_idx = backup_last_idx

                if is_forced_cut:
                    cut_point = words[last_word_idx]['end']
                    end_buffer = 0 
                else:
                    cut_point = (words[last_word_idx]['end'] + words[last_word_idx+1]['start']) / 2
                    end_buffer = BUFFER_SEC

                chunk_words = words[curr_idx : last_word_idx + 1]
                chunk_text = align_punctuation(full_text, chunk_words)
                
                s_smp = int(max(0, start_sec - BUFFER_SEC) * sr)
                e_smp = int(min(total_samples / sr, cut_point + end_buffer) * sr)
                chunk_wave = waveform[:, s_smp:e_smp]

                if chunk_wave.shape[1] > 0:
                    out_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{split_idx}.wav"
                    out_p = os.path.join(OUTPUT_DIR, out_name)
                    save_safe_chunk(chunk_wave, chunk_text, out_p, sr, final_entries, is_forced_cut)
                
                curr_idx = last_word_idx + 1
                split_idx += 1
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    with open(FINAL_MANIFEST, 'w', encoding='utf-8') as f:
        for e in final_entries: 
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run_slicing_pipeline()