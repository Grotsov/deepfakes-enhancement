import json
import re

def normalize_and_map(raw_text):
    """Normalizes text for mapping while preserving original string indices."""
    norm_chars, index_map = [], []
    for i, char in enumerate(raw_text):
        if char.isalnum():
            norm_chars.append(char.lower())
            index_map.append(i)
    return "".join(norm_chars), index_map

def generate_smart_review(manifest_path, book_txt_path):
    """Validates audio segments against the source text for quote fragmentation."""
    with open(book_txt_path, 'r', encoding='utf-8') as f:
        raw_book = f.read()

    # Build index map
    norm_book, index_map = normalize_and_map(raw_book)
    
    insertions = []
    last_found_index = 0
    search_buffer = 100000

    safe_items = []
    not_found_items = []
    check_map = {} 
    short_id_counter = 1

    # Analyze quote context
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                transcript = item.get('text_normalized', item.get('text', ''))
                norm_transcript, _ = normalize_and_map(transcript)
                
                if not norm_transcript: continue

                # Search within text buffer
                search_window = norm_book[last_found_index : last_found_index + search_buffer]
                match_idx = search_window.find(norm_transcript)
                
                if match_idx != -1: 
                    start_norm_idx = last_found_index + match_idx
                else:
                    match_idx = norm_book.find(norm_transcript)
                    if match_idx != -1: 
                        start_norm_idx = match_idx
                    else:
                        # Flag if not found in the source text
                        item['reason'] = 'not_found_in_text'
                        not_found_items.append(item)
                        continue

                end_norm_idx = start_norm_idx + len(norm_transcript) - 1
                last_found_index = start_norm_idx

                orig_start = index_map[start_norm_idx]
                orig_end = index_map[end_norm_idx]

                # Paragraph highlighting for contextual boundary analysis
                para_start = raw_book.rfind('\n\n', 0, orig_start)
                if para_start == -1: para_start = 0
                para_end = raw_book.find('\n\n', orig_end)
                if para_end == -1: para_end = len(raw_book)

                text_before = raw_book[para_start:orig_start]
                text_audio = raw_book[orig_start:orig_end+1]
                text_after = raw_book[orig_end+1:para_end]

                is_questionable = False

                # Dialogue analysis logic: Check for orphaned quotation marks
                # 1. Audio reference contains quotes
                if any(q in text_audio for q in ['"', '“', '”', '«', '»', '—']):
                    is_questionable = True
                    
                # 2. Audio reference inside quotes in the original text
                elif text_before.count('"') % 2 != 0:
                    is_questionable = True
                elif text_before.count('“') > text_before.count('”') or text_before.count('«') > text_before.count('»'):
                    is_questionable = True    

                if is_questionable:
                    check_map[str(short_id_counter)] = item
                    insertions.append((orig_start, f"[00 {short_id_counter}] "))
                    insertions.append((orig_end + 1, " [/]"))
                    short_id_counter += 1
                else:
                    safe_items.append(item)

            except json.JSONDecodeError: continue

    print(f"Safe refs: {len(safe_items)}\n- Refs missing: {len(not_found_items)}\n- Refs to check: {len(check_map)}")

    # 1. Save safe references
    with open("manifest_safe.json", "w", encoding="utf-8") as f:
        for item in safe_items: f.write(json.dumps(item) + "\n")
        
    # 2. Save missing references
    with open("manifest_not_found.json", "w", encoding="utf-8") as f:
        for item in not_found_items: f.write(json.dumps(item) + "\n")

    # 3. Save check map for manual review
    with open("check_map.json", "w", encoding="utf-8") as f:
        json.dump(check_map, f, ensure_ascii=False, indent=2)

    # 4. Create annotated review file
    insertions.sort(key=lambda x: x[0], reverse=True)
    annotated_book = raw_book
    for idx, text_to_insert in insertions:
        annotated_book = annotated_book[:idx] + text_to_insert + annotated_book[idx:]

    paragraphs = annotated_book.split('\n\n')
    review_paragraphs = [p.strip() for p in paragraphs if "[00 " in p]

    with open("review_book.txt", "w", encoding="utf-8") as f:
        f.write("\n\n" + ("="*40) + "\n\n".join(review_paragraphs))

if __name__ == "__main__":
    # Execute extraction for the specific D'Artagnan dataset
    generate_smart_review("dartagnan01.json", "dartagnan01.txt")