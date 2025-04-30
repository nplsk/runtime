# assign_phases.py
import os
import json

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"
OUTPUT_FILE = "movement_archive.json"

phase_assignments = {
    "orientation": [],
    "elemental": [],
    "built": [],
    "people": [],
    "blur": []
}

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        semantic_tags = set(data.get("semantic_tags", []))
        formal_tags = set(data.get("formal_tags", []))
        emotional_tags = set(data.get("emotional_tags", []))
        material_tags = set(data.get("material_tags", []))
        mood_tag = data.get("mood_tag", "")
        motion_tag = data.get("motion_tag", "")
        motion_score = data.get("motion_score", 0)
        all_tags = semantic_tags | formal_tags | emotional_tags | material_tags

        filename = os.path.basename(filepath)

        # Always include in 'orientation'
        phase_assignments["orientation"].append(filename)

        # ELEMENTAL
        elemental_tags = {
            "nature", "forest", "water", "snow", "serene", "calm", "still",
            "ambient_light", "diffused_light", "natural_light", "grassy_field",
            "rocks", "earth_materials", "vegetation", "fog", "trees", "stream", "wildlife", "sky"
        }
        if elemental_tags & all_tags or mood_tag in {"cool", "neutral"}:
            if motion_tag in {"still", "moderate"} or motion_score <= 12:
                phase_assignments["elemental"].append(filename)
                continue

        # BUILT
        built_tags = {
            "interior", "architecture", "urban", "structure", "object", "enclosed_space",
            "room", "bookshelves", "chairs", "sofas", "living_room", "apartment", "furniture", "kitchen"
        }
        if built_tags & all_tags and motion_score >= 8:
            phase_assignments["built"].append(filename)
            continue

        # PEOPLE
        people_tags = {
            "group", "portrait", "audience", "hands", "family", "figures",
            "individuals", "people", "musicians", "birthday_cake", "celebration",
            "dance", "performance", "duo", "crowd"
        }
        if people_tags & all_tags:
            phase_assignments["people"].append(filename)
            continue

        # BLUR
        blur_tags = {
            "chaotic", "dynamic", "blur", "fast", "motion_blur", "shaky",
            "rapid_motion", "blurry", "high-energy", "vibrant"
        }
        if motion_tag == "dynamic" or motion_score > 30 or blur_tags & all_tags:
            phase_assignments["blur"].append(filename)
            continue

        # If no phase matched explicitly, we still count it as orientation
        if filename not in phase_assignments["elemental"] + \
                        phase_assignments["built"] + \
                        phase_assignments["people"] + \
                        phase_assignments["blur"]:
            phase_assignments["orientation"].append(filename)
            print(f"⚠️ Defaulted to 'orientation': {filename}")

    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Save assignment
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(phase_assignments, f, indent=2)

print(f"✅ Saved movement archive to {OUTPUT_FILE}")