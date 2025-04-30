# normalize_tags.py
import os
import json

OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"

TAG_CANONICAL = {
    # Existing
    "warm_light": "warm_lighting",
    "black_and_white": "monochrome",
    "grayscale": "monochrome",
    "monochromatic_image": "monochrome",
    "soft_light": "soft_lighting",
    "stillness": "static",

    # New Suggestions
    "blue_light": "cool_lighting",
    "cool_tones": "cool_lighting",
    "cool_tone": "cool_lighting",
    "warm_colors": "warm_lighting",
    "warm_glow": "warm_lighting",
    "warmth": "warm_lighting",
    "lighting_conditions": "lighting",
    "diffused_lighting": "soft_lighting",
    "ambient_light": "soft_lighting",
    "sunlight": "natural_light",
    "daylight": "natural_light",
    "night": "low_light",
    "fog": "diffused_light",
    "motion_blur": "motion",
    "dynamic_motion": "motion",
    "moderate_motion": "motion",
    "still": "static",
    "animated": "motion",
    "dance_floor": "interior",
    "seating": "furniture",
    "chair": "furniture",
    "couches": "furniture",
    "sofa": "furniture",
    "cool_colored": "cool_lighting",
    "cool_colors": "cool_lighting",
    "cool_mood": "cool",
    "warm": "warm_lighting",
    "darkness": "low_light",
    "black_and_white_dog": "dog",
    "color_mood": "color",
}
def normalize_tag_list(tag_list):
    return list(sorted(set(
        TAG_CANONICAL.get(t.strip().lower().replace(" ", "_"), t.strip().lower().replace(" ", "_"))
        for t in tag_list if isinstance(t, str)
    )))

MATERIAL_BANLIST = {"object", "thing", "item", "element", "structure", "stuff", "shape", "scene"}

for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
            if key in data:
                cleaned = normalize_tag_list(data[key])
                if key == "material_tags":
                    cleaned = [tag for tag in cleaned if tag not in MATERIAL_BANLIST]
                data[key] = cleaned

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Normalized {filename}")
    except Exception as e:
        print(f"Failed normalizing {filename}: {e}")