import json
import os
from pathlib import Path

# ================= SETTINGS =================
TERMINAL_LIST = "terminal_casualties_to_delete.json"
DIRECTORIES_TO_CLEAN = [
    "recordings",
    "mask_gct",
    "fish_speech",
    "f5tts",
    "cosy_voice"
]

def main():
    target_list_path = Path(TERMINAL_LIST)
    if not target_list_path.exists():
        print(f"Error: {TERMINAL_LIST} not found.")
        return

    with open(target_list_path, 'r', encoding='utf-8') as f:
        blacklist = set(json.load(f))

    print(f"Unique files identified for global purge: {len(blacklist)}\n")
    total_deleted = 0

    for directory in DIRECTORIES_TO_CLEAN:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"[SKIPPED] Directory not found: {directory}")
            continue
            
        print(f"Cleaning directory: {directory} ...")
        deleted_in_dir = 0
        
        for wav_file in dir_path.rglob('*.wav'):
            if wav_file.name in blacklist:
                try:
                    wav_file.unlink() 
                    deleted_in_dir += 1
                    total_deleted += 1
                except Exception as e:
                    print(f"Error deleting {wav_file.name}: {e}")
                    
        print(f" -> Removed: {deleted_in_dir} files")

    print("\n" + "="*40)
    print(f"SUMMARY: Physically purged {total_deleted} .wav files across all datasets.")
    print("="*40)

if __name__ == "__main__":
    main()