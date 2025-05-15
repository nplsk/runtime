import json
import os
import random
from datetime import timedelta

# Movement timing structure
MOVEMENTS = [
    {"name": "orientation", "min_dur": 20, "max_dur": 45, "clip_count": 1, "tags": []},
    {"name": "elemental", "min_dur": 15, "max_dur": 40, "clip_count": 2, "tags": ["nature", "stillness"]},
    {"name": "built", "min_dur": 10, "max_dur": 30, "clip_count": 2, "tags": ["architecture", "urban", "empty"]},
    {"name": "people", "min_dur": 5, "max_dur": 20, "clip_count": 3, "tags": ["people", "crowd", "family"]},
    {"name": "blur", "min_dur": 1, "max_dur": 6, "clip_count": 4, "tags": ["motion", "blurry", "abstract"]},
]

# Tag weight modifiers (optional tuning)
TAG_WEIGHTS = {
    "semantic_tags": 1.0,
    "formal_tags": 0.75,
    "emotional_tags": 0.5
}

# Scoring parameters
MOTION_THRESHOLD = 0.4
VARIANCE_THRESHOLD = 0.3

# Path to JSON scene metadata and movement archive
SCENE_FOLDER = "./scene_metadata"
MOVEMENT_MAP_PATH = "./movement_archive.json"

# Load movement archive mapping
with open(MOVEMENT_MAP_PATH) as f:
    movement_map = json.load(f)

# Load all scene metadata
scene_data = {}
for filename in os.listdir(SCENE_FOLDER):
    if filename.endswith(".json"):
        with open(os.path.join(SCENE_FOLDER, filename)) as f:
            data = json.load(f)
            scene_data[filename] = data

# Helper: Get all tags in a flat list
def get_all_tags(scene):
    tags = []
    for k in TAG_WEIGHTS:
        tags.extend(scene.get(k, []))
    return tags

# Helper: Score similarity by matching tags

def tag_score(tags1, tags2):
    return len(set(tags1) & set(tags2)) / max(1, len(set(tags2)))

# Build score timeline
def build_score():
    timeline = []
    for move in MOVEMENTS:
        candidates = [
            (fname, data) for fname, data in scene_data.items()
            if movement_map.get(fname) == move["name"]
        ]
        
        clips = []
        random.shuffle(candidates)

        for fname, data in candidates:
            if len(clips) >= move["clip_count"]:
                break

            score = tag_score(get_all_tags(data), move["tags"])
            if score > 0.2 or not move["tags"]:
                dur = random.uniform(move["min_dur"], move["max_dur"])
                clips.append({
                    "filename": fname,
                    "duration": round(dur, 1),
                    "score": round(score, 2),
                    "motion": data.get("motion_score"),
                    "variance": data.get("motion_variance"),
                    "tags": get_all_tags(data)
                })

        timeline.append({
            "movement": move["name"],
            "clips": clips
        })

    return timeline

if __name__ == "__main__":
    score = build_score()
    with open("generated_score.json", "w") as out:
        json.dump(score, out, indent=2)
    print("✅ Sequence score written to generated_score.json")
