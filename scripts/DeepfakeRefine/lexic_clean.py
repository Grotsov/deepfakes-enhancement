import json
import re
import argparse
from whisper.normalizers import EnglishTextNormalizer
import jiwer

# Initialize normalizer and dictionaries
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

def run_stage2_salvage(model_name):
    input_file = f"{model_name}_mismatched.json"
    output_salvaged = f"{model_name}_salvaged.json"
    output_true_bad = f"{model_name}_true_bad.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    salvaged, true_bad = [], []
    
    for entry in data:
        exp = entry.get('expected_raw', '')
        trn = entry.get('transcribed_raw', '')
        
        # Immediate acoustic rejection
        if any("CONF" in r for r in entry.get('reasons', [])):
            true_bad.append(entry)
            continue
            
        norm_exp = custom_clean(exp)
        norm_trn = custom_clean(trn)
        
        new_wer = jiwer.wer(norm_exp, norm_trn)
        
        exp_words = norm_exp.split()
        trn_words = norm_trn.split()
        head_len = min(2, len(exp_words), len(trn_words))
        is_bad_head = exp_words[:head_len] != trn_words[:head_len] if head_len > 0 else True
        is_exact_no_space = norm_exp.replace(" ", "") == norm_trn.replace(" ", "")
        
        if (new_wer <= 0.05 and not is_bad_head) or is_exact_no_space:
            entry['new_wer'] = round(new_wer, 3) if not is_exact_no_space else 0.0
            salvaged.append(entry)
        else:
            entry['new_wer'] = round(new_wer, 3)
            true_bad.append(entry)

    with open(output_salvaged, 'w', encoding='utf-8') as f:
        json.dump(salvaged, f, indent=4, ensure_ascii=False)
    with open(output_true_bad, 'w', encoding='utf-8') as f:
        json.dump(true_bad, f, indent=4, ensure_ascii=False)
        
    print(f"[{model_name.upper()}] Stage 2 Complete.")
    print(f"Total processed: {len(data)} | Salvaged: {len(salvaged)} | True Bad: {len(true_bad)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["maskgct", "f5tts", "fishs1", "cosyvoice2"])
    args = parser.parse_args()
    run_stage2_salvage(args.model)