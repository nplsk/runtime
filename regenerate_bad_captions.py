# regenerate_bad_captions.py
import os, json
from process_videos import process_video
from tqdm import tqdm

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED"

def is_bad_caption(caption):
    if not caption or len(caption.strip().split()) < 5:
        return True
    lowered = caption.lower()
    return lowered.startswith("no caption") or "screenshot" in lowered or "image of" in lowered

bad_files = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".json") or file.startswith("._"):
        continue
    try:
        with open(os.path.join(INPUT_DIR, file), "r") as f:
            data = json.load(f)
        if is_bad_caption(data.get("semantic_caption", "")):
            bad_files.append(data["file_path"])
    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"Found {len(bad_files)} bad captions. Reprocessing...")

for path in tqdm(bad_files):
    try:
        process_video(path)
    except Exception as e:
        print(f"Failed to process {path}: {e}")