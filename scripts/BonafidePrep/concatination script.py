import json
import os
from collections import defaultdict

# Configuration Parameters
INPUT_MANIFEST = "manifest_prepared.json"
OUTPUT_JSONL = "merge_plan.jsonl"
MIN_DUR = 5.0
MAX_DUR = 10.0

def smart_stitch_manifest(manifest_path, output_jsonl, min_dur=5.0, max_dur=10.0):
    """Constructs a composite metadata plan for audio concatenation."""
    groups = defaultdict(list)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except ValueError: 
                continue
            
            filepath = data.get("audio_filepath", "")
            dir_name = os.path.dirname(filepath)
            filename = os.path.splitext(os.path.basename(filepath))[0]
            
            # Isolate the sequence number from the file nomenclature
            parts = filename.split('_')
            if len(parts) < 2: continue
            
            base_name = "_".join(parts[:-1])
            try:
                seq_num = int(parts[-1])
            except ValueError:
                continue
            
            group_key = (dir_name, base_name)
            data['seq_num'] = seq_num
            groups[group_key].append(data)

    valid_results = []
    
    # Heuristic Batching Logic
    for (dir_name, base_name), items in groups.items():
        items.sort(key=lambda x: x['seq_num'])
        
        current_batch = []
        current_dur = 0.0
        
        for item in items:
            gap = current_batch and (item['seq_num'] != current_batch[-1]['seq_num'] + 1)
            too_long = current_batch and (current_dur + item['duration'] > max_dur)
            
            if gap or too_long:
                if current_dur >= min_dur:
                    valid_results.append(current_batch)
                current_batch = []
                current_dur = 0.0
            
            current_batch.append(item)
            current_dur += item['duration']
            
        if current_batch and current_dur >= min_dur:
            valid_results.append(current_batch)

    # Metadata Aggregation and JSONL Generation
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for batch in valid_results:
            if len(batch) == 1:
                entry = batch[0].copy()
                entry.pop('seq_num', None)
                entry['source_files'] = [entry['audio_filepath']]
            else:
                first, last = batch[0], batch[-1]
                
                dir_name = os.path.dirname(first['audio_filepath'])
                filename = os.path.splitext(os.path.basename(first['audio_filepath']))[0]
                base_name = "_".join(filename.split('_')[:-1])
                
                # Generate contiguous range identifier
                file_id = f"{base_name}_{first['seq_num']}_{last['seq_num']}"
                new_path = os.path.join(dir_name, f"{file_id}.wav").replace("\\", "/")
                
                texts = []
                for x in batch:
                    t = x.get('text_normalized') or x.get('text') or ""
                    texts.append(t.strip())
                
                entry = first.copy()
                entry.pop('seq_num', None)
                entry.update({
                    "audio_filepath": new_path,
                    "text": " ".join(texts),
                    "duration": round(sum([x['duration'] for x in batch]), 2),
                    "source_files": [x['audio_filepath'] for x in batch]
                })
            
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    smart_stitch_manifest(INPUT_MANIFEST, OUTPUT_JSONL, MIN_DUR, MAX_DUR)