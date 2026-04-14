import json
import random

# ================= SETTINGS =================
INPUT_MANIFEST = "parallel_master_manifest.jsonl" 

def split_manifest(input_path, train_p=0.9, val_p=0.05, test_p=0.05):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Deterministic global shuffle for reproducibility
    random.seed(42)
    random.shuffle(lines)

    total = len(lines)
    train_idx = int(total * train_p)
    val_idx = train_idx + int(total * val_p)

    train_data = lines[:train_idx]
    val_data = lines[train_idx:val_idx]
    test_data = lines[val_idx:]

    with open('train.jsonl', 'w') as f: f.writelines(train_data)
    with open('val.jsonl', 'w') as f: f.writelines(val_data)
    with open('test.jsonl', 'w') as f: f.writelines(test_data)

    print(f"Data Partitioning Complete! Total parallel arrays: {total}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    split_manifest(INPUT_MANIFEST)