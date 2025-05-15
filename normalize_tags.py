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

    "natural_lighting": "natural_light",
    "blue_lighting": "cool_lighting",
    "cool-toned_lighting": "cool_lighting",
    "dim_light": "low_light",
    "ambient_lighting": "soft_lighting",
    "balanced_lighting": "soft_lighting",
    "warm_color_palette": "warm_lighting",
    "warm_color_mood": "warm_lighting",
    "orange_light": "warm_lighting",
    "black_and_white_image": "monochrome",
    "black_and_white_photo": "monochrome",
    "black_and_white_photograph": "monochrome",
    "monochromatic": "monochrome",
    "close-up": "close_up",
    "close-up_object": "close_up",
    "standing_person": "standing",
    "person": "human",
    "man": "human",
    "woman": "human",
    "people": "group",
    "men": "group",
    "women": "group",
    "group_of_people": "group",
    "person_lying": "laying",
    "lying_down": "laying",
    "laying": "laying",
    "sofas": "furniture",
    "chairs": "furniture",
    "couch": "furniture",
    "black_couch": "furniture",
    "table": "furniture",
    "desk": "furniture",
    "tables": "furniture",
    "bed": "furniture",
    "musicians": "performer",
    "musician": "performer",
    "jazz_band": "performer",
    "band": "performer",
    "performance": "event",
    "concert": "event",
    "live_music": "event",
    "live_performance": "event",
    "music_performance": "event",
    "ice_crystals": "ice",
    "ice_patterns": "ice",
    "icy": "ice",
    "frost": "ice",
    "sleet": "snow",
    "snowflakes": "snow",
    "wintry": "snow",

    "dimly_lit": "low_light",
    "overhead": "high_angle",
    "tilted": "dynamic_angle",
    "obscured": "abstract",
    "blurry": "abstract",
    "parking_lot": "built_environment",
    "bedroom": "interior",
    "studio": "interior",
    "snowy_landscape": "snow",
    "natural_color": "natural_light",
    "crowd": "group",
    "audience": "group",
    "hardwood_floor": "floor",
    "wooden_floor": "floor",
    "wood_floors": "floor",
    "polished_floor": "floor",
    "concrete_floor": "floor",
    "forest_floor": "forest",
    "wooded_forest": "forest",
    "dj_app": "dj_booth",
    "dj_setup": "dj_booth",
    "equipment_dj": "dj_booth",
    "dj_equipment": "dj_booth",
    "fabric_or_clothes": "clothes",
    "fabric(clothes)": "clothes",
    "fabric_gowns": "clothes",
    "the_glow_of_the_screen": "glow",
    "boyfriend": "human",
    "dreamlike": "surreal",
    "metallic_instruments": "instruments",
    "cool_toned_lighting": "cool_lighting",
    "warm_toned_lightings": "warm_lighting",
    "gold_tones": "natural_tones",
    "amber_tones": "natural_tones",
    "monochromatic_tone": "monochromatic",
    "shimmer": "reflections",
    "shimmering": "reflections",
    "light_effects": "reflections"
}

def clean_tag_name(tag):
    tag = tag.lower().replace(" ", "_")
    for suffix in ["_photo", "_image", "_view", "_snapshot", "_display", "_object", "_scene", "_background"]:
        if tag.endswith(suffix):
            tag = tag.removesuffix(suffix)
    return TAG_CANONICAL.get(tag, tag)

def normalize_tag_list(tag_list):
    return list(sorted(set(
        clean_tag_name(t) for t in tag_list if isinstance(t, str)
    )))

MATERIAL_BANLIST = {
    "object", "thing", "item", "element", "structure", "stuff",
    "image", "images", "photo", "view", "display", "visual", "snapshot",
    "scene", "composition", "background", "foreground", "picture", "visuals",
    "eczema_diagnosis", "color_tones"
}

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