import json
import re
import argparse
from whisper.normalizers import EnglishTextNormalizer
import jiwer

whisper_norm = EnglishTextNormalizer()

NAME_FIXES = {
    "bessy": "bessie", "bessys": "bessies", "earle": "earl",
    "jem": "gem", "maurice": "morris", "martial": "marshall",
    "athos": "aethos", "face-of-god": "face of god",
    "folk-might": "folkmite", "stone-face": "stoneface", "whiles": "whilst"
}

ARCHAIC_FIXES = {
    "wherean": "wherein", "where ain't": "wherein",
    "how bait": "howbeit", "wiles": "whiles",
    "nought": "naught", "oclock": "o'clock"
}

def custom_clean(text):
    if not text: return ""
    text = text.lower()
    text = text.replace("'s", "s").replace("'", "")
    
    for old, new in NAME_FIXES.items():
        text = re.sub(rf'\b{old}\b', new, text)
    for old, new in ARCHAIC_FIXES.items():
        text = re.sub(rf'\b{old}\b', new, text)
        
    return whisper_norm(text)

def run_stage5_profiling(model_name):
    input_file = f"{model_name}_bad_mismatch.json" # Input from Stage 4
    output_salvaged = f"{model_name}_salvaged_pass2.json"
    output_revision = f"{model_name}_bad_revision.json"
    output_arch_failure = f"{model_name}_arch_failure.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    salvaged, bad_revision, arch_failures = [], [], []
    
    for entry in data:
        exp = entry.get('expected_raw', '')
        trn = entry.get('transcribed_raw', '')
        
        if any("CONF" in r for r in entry.get('reasons', [])):
            bad_revision.append(entry)
            continue
            
        norm_exp = custom_clean(exp)
        norm_trn = custom_clean(trn)
        new_wer = jiwer.wer(norm_exp, norm_trn)
        entry['new_wer'] = round(new_wer, 3)
        
        exp_words = norm_exp.split()
        trn_words = norm_trn.split()
        
        # Base logical checks
        head_len = min(2, len(exp_words), len(trn_words))
        is_bad_head = exp_words[:head_len] != trn_words[:head_len] if head_len > 0 else True
        is_exact_no_space = norm_exp.replace(" ", "") == norm_trn.replace(" ", "")
        
        # 1. Salvage check
        if (new_wer <= 0.05 and not is_bad_head) or is_exact_no_space:
            salvaged.append(entry)
            continue

        # 2. ARCHITECTURE SPECIFIC PROFILING
        if model_name == "maskgct":
            # Heuristic: Prefix Hallucinations
            is_prefix_hallucination = False
            if is_bad_head or len(trn_words) > len(exp_words):
                for shift in range(1, 4):
                    if len(trn_words) > shift:
                        shifted_trn = " ".join(trn_words[shift:])
                        if jiwer.wer(norm_exp, shifted_trn) <= 0.10:
                            is_prefix_hallucination = True
                            entry['failure_type'] = "prefix_hallucination"
                            break
            if is_prefix_hallucination: arch_failures.append(entry)
            else: bad_revision.append(entry)

        elif model_name == "fishs1":
            # Heuristic: Early End-of-Sentence (EOS)
            is_early_eos = False
            if len(exp_words) >= 3:
                dropped_length = len(exp_words) - len(trn_words) >= 3
                missed_tail_words = exp_words[-3:] != trn_words[-3:] if len(trn_words) >= 3 else True
                if dropped_length and missed_tail_words: is_early_eos = True
            elif len(exp_words) > len(trn_words):
                is_early_eos = True
                
            if new_wer > 0.15 and is_early_eos:
                entry['failure_type'] = "early_eos"
                arch_failures.append(entry)
            else: bad_revision.append(entry)

        elif model_name == "cosyvoice2":
            # Heuristic: Strict 1-word Head Skip
            is_bad_head_strict = exp_words[:1] != trn_words[:1] if len(exp_words) > 0 and len(trn_words) > 0 else True
            if is_bad_head_strict:
                entry['failure_type'] = "head_skip"
                arch_failures.append(entry)
            else: bad_revision.append(entry)

        elif model_name == "f5tts":
            # F5TTS lacked a specific heuristic structural flaw, routed to manual audit
            bad_revision.append(entry)

    # Save outputs
    with open(output_salvaged, 'w', encoding='utf-8') as f: json.dump(salvaged, f, indent=4)
    with open(output_revision, 'w', encoding='utf-8') as f: json.dump(bad_revision, f, indent=4)
    with open(output_arch_failure, 'w', encoding='utf-8') as f: json.dump(arch_failures, f, indent=4)

    print(f"[{model_name.upper()}] Stage 5 Profiling Complete.")
    print(f"Salvaged: {len(salvaged)} | Known Arch Failures: {len(arch_failures)} | Manual Revision: {len(bad_revision)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["maskgct", "f5tts", "fishs1", "cosyvoice2"])
    args = parser.parse_args()
    run_stage5_profiling(args.model)