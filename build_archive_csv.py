import os
import json
import csv
from glob import glob
from tqdm import tqdm  # for pretty printing progress

# Config
SIDECAR_DIR = "./output"  # Folder containing .json sidecars
OUTPUT_CSV = "./archive.csv"
VIDEO_DIR = "./scenes"  # Optional: can be used to auto-fill paths

# Fields to extract
FIELDS = [
    "video_id",
    "file_path",
    "duration",
    "cluster",
    "cluster_name",
    "semantic_tags",
    "formal_tags",
    "emotional_tags",
    "preliminary_tags",
    "time_tags",
    "material_tags",
    "dominant_colors",
    "semantic_caption",
    "ai_description",
    # "segments"
]

def load_sidecar(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_semantic_caption(caption):
    if caption:
        return caption.replace('|', ' ').strip()
    return ""

def format_list(value):
    if isinstance(value, list):
        return ', '.join(value)
    return value or ""

def build_archive():
    entries = []
    sidecar_files = glob(os.path.join(SIDECAR_DIR, "*.json"))

    for json_file in tqdm(sidecar_files, desc="Building archive CSV", ncols=80):
        data = load_sidecar(json_file)
        entry = {}

        for field in FIELDS:
            if field in data:
                if field.endswith('_tags') or field == 'dominant_colors':
                    entry[field] = format_list(data[field])
                elif field == 'semantic_caption':
                    entry[field] = clean_semantic_caption(data[field])
                else:
                    entry[field] = data[field]
            else:
                entry[field] = ""

        # Optional: Auto-fill file_path if missing
        if not entry['file_path'] and 'video_id' in entry:
            entry['file_path'] = os.path.join(VIDEO_DIR, f"{entry['video_id']}.mov")

        entries.append(entry)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(entries)

    print(f"Archive CSV written to {OUTPUT_CSV} with {len(entries)} entries.")

if __name__ == "__main__":
    build_archive()
