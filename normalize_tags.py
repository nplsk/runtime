"""
This script normalizes and standardizes tags in video metadata JSON files.
It processes semantic, formal, emotional, and material tags to ensure consistency
across the video database by:
- Converting tags to a canonical form
- Removing redundant or generic terms
- Standardizing tag formatting
- Filtering out banned material tags

The script maintains a mapping of alternative tag forms to their canonical versions
and applies various cleaning rules to improve tag quality and consistency.
"""

import os
import json

OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"

# Mapping of alternative tag forms to their canonical versions
TAG_CANONICAL = {
    # Lighting-related tags
    "warm_light": "warm_lighting",
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
    "dim_light": "low_light",
    "ambient_lighting": "soft_lighting",
    "balanced_lighting": "soft_lighting",
    "warm_color_palette": "warm_lighting",
    "warm_color_mood": "warm_lighting",
    "orange_light": "warm_lighting",
    "natural_lighting": "natural_light",
    "blue_lighting": "cool_lighting",
    "cool-toned_lighting": "cool_lighting",
    "dimly_lit": "low_light",
    "cool_toned_lighting": "cool_lighting",
    "warm_toned_lightings": "warm_lighting",

    # Color and tone tags
    "black_and_white": "monochrome",
    "grayscale": "monochrome",
    "monochromatic_image": "monochrome",
    "black_and_white_image": "monochrome",
    "black_and_white_photo": "monochrome",
    "black_and_white_photograph": "monochrome",
    "monochromatic": "monochrome",
    "monochromatic_tone": "monochromatic",
    "gold_tones": "natural_tones",
    "amber_tones": "natural_tones",

    # Motion-related tags
    "stillness": "static",
    "motion_blur": "motion",
    "dynamic_motion": "motion",
    "moderate_motion": "motion",
    "still": "static",
    "animated": "motion",

    # Furniture and interior tags
    "dance_floor": "interior",
    "seating": "furniture",
    "chair": "furniture",
    "couches": "furniture",
    "sofa": "furniture",
    "sofas": "furniture",
    "chairs": "furniture",
    "couch": "furniture",
    "black_couch": "furniture",
    "table": "furniture",
    "desk": "furniture",
    "tables": "furniture",
    "bed": "furniture",
    "bedroom": "interior",
    "studio": "interior",

    # Human-related tags
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
    "crowd": "group",
    "audience": "group",
    "boyfriend": "human",

    # Performance-related tags
    "musicians": "performer",
    "musician": "performer",
    "jazz_band": "performer",
    "band": "performer",
    "performance": "event",
    "concert": "event",
    "live_music": "event",
    "live_performance": "event",
    "music_performance": "event",

    # Nature and weather tags
    "ice_crystals": "ice",
    "ice_patterns": "ice",
    "icy": "ice",
    "frost": "ice",
    "sleet": "snow",
    "snowflakes": "snow",
    "wintry": "snow",
    "snowy_landscape": "snow",
    "forest_floor": "forest",
    "wooded_forest": "forest",

    # Environment and location tags
    "parking_lot": "built_environment",
    "hardwood_floor": "floor",
    "wooden_floor": "floor",
    "wood_floors": "floor",
    "polished_floor": "floor",
    "concrete_floor": "floor",

    # Equipment and technical tags
    "dj_app": "dj_booth",
    "dj_setup": "dj_booth",
    "equipment_dj": "dj_booth",
    "dj_equipment": "dj_booth",
    "metallic_instruments": "instruments",

    # Other descriptive tags
    "fabric_or_clothes": "clothes",
    "fabric(clothes)": "clothes",
    "fabric_gowns": "clothes",
    "the_glow_of_the_screen": "glow",
    "dreamlike": "surreal",
    "shimmer": "reflections",
    "shimmering": "reflections",
    "light_effects": "reflections"
}

def clean_tag_name(tag):
    """
    Clean and standardize a single tag name.
    
    Args:
        tag: The tag string to clean
        
    Returns:
        Cleaned and canonicalized tag name
    """
    # Convert to lowercase and replace spaces with underscores
    tag = tag.lower().replace(" ", "_")
    
    # Remove common redundant suffixes
    for suffix in ["_photo", "_image", "_view", "_snapshot", "_display", "_object", "_scene", "_background"]:
        if tag.endswith(suffix):
            tag = tag.removesuffix(suffix)
    
    # Return canonical form if it exists, otherwise return cleaned tag
    return TAG_CANONICAL.get(tag, tag)

def normalize_tag_list(tag_list):
    """
    Normalize a list of tags by cleaning each tag and removing duplicates.
    
    Args:
        tag_list: List of tag strings to normalize
        
    Returns:
        Sorted list of unique, normalized tags
    """
    return list(sorted(set(
        clean_tag_name(t) for t in tag_list if isinstance(t, str)
    )))

# Set of generic or unhelpful material tags to filter out
MATERIAL_BANLIST = {
    "object", "thing", "item", "element", "structure", "stuff",
    "image", "images", "photo", "view", "display", "visual", "snapshot",
    "scene", "composition", "background", "foreground", "picture", "visuals",
    "eczema_diagnosis", "color_tones"
}

# Process all JSON files in the output directory
for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        # Load and process each JSON file
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize tags in each category
        for key in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
            if key in data:
                cleaned = normalize_tag_list(data[key])
                # Additional filtering for material tags
                if key == "material_tags":
                    cleaned = [tag for tag in cleaned if tag not in MATERIAL_BANLIST]
                data[key] = cleaned

        # Save normalized data back to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Normalized {filename}")
    except Exception as e:
        print(f"Failed normalizing {filename}: {e}")