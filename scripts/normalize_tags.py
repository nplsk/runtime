"""
This script normalizes and standardizes tags in video metadata JSON files.
It processes semantic, formal, emotional, and material tags to ensure consistency
across the video database by:
- Converting tags to a canonical form
- Removing redundant or generic terms
- Standardizing tag formatting
- Filtering out banned material tags

The script also generates a report of tag inconsistencies before normalization.
"""

import os
import json
import sys
import re
from pathlib import Path
from collections import defaultdict

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config import OUTPUT_DIR

def report_tag_inconsistencies():
    """
    Generate a report of tag inconsistencies across all JSON files.
    This includes plural/singular, case/character normalization, and potential synonyms.
    """
    print("\nGenerating tag inconsistency report...")
    tag_frequency = defaultdict(int)
    tag_variants = defaultdict(set)
    tag_categories = defaultdict(set)
    tag_files = defaultdict(list)

    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.json') and not filename.startswith('._'):
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('metadata_payload', {})
                for key in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
                    if key in metadata:
                        for tag in metadata[key]:
                            tag_frequency[tag] += 1
                            tag_variants[tag.lower()].add(tag)
                            tag_categories[tag].add(key)
                            tag_files[tag].append(filename)

    print("\nTag Inconsistency Report:")
    print("------------------------")
    print("1. Plural/Singular Inconsistencies:")
    for tag, variants in tag_variants.items():
        if len(variants) > 1:
            print(f"  {tag}: {variants}")

    print("\n2. Case/Character Inconsistencies:")
    for tag, variants in tag_variants.items():
        if len(variants) > 1:
            print(f"  {tag}: {variants}")

    print("\n3. Tags with Special Characters or Numbers:")
    for tag in tag_frequency:
        if re.search(r'[^a-z_]', tag):
            print(f"  {tag}")

    print("\n4. Tags with Low Frequency (Potential Typos):")
    for tag, freq in tag_frequency.items():
        if freq < 2:
            print(f"  {tag} (frequency: {freq})")

    print("\n5. Tags with High Frequency (Potentially Generic):")
    for tag, freq in tag_frequency.items():
        if freq > 10:
            print(f"  {tag} (frequency: {freq})")

    print("\n6. Tags with Multiple Categories:")
    for tag, categories in tag_categories.items():
        if len(categories) > 1:
            print(f"  {tag}: {categories}")

    print("\n7. Tags with Multiple Files:")
    for tag, files in tag_files.items():
        if len(files) > 1:
            print(f"  {tag}: {files}")

    print("\nTag Inconsistency Report Complete!")
    return tag_frequency, tag_variants, tag_categories

# Mapping of alternative tag forms to their canonical versions
TAG_CANONICAL = {
    # Fix truncated tags
    "gla": "glass",
    "stillne": "stillness",
    "darkne": "darkness",
    "calmne": "calmness",
    "mysteriou": "mysterious",
    "ominou": "ominous",
    
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
    
    # Clean up special characters and multiple underscores
    tag = re.sub(r'[^a-z0-9_]', '_', tag)
    tag = re.sub(r'_+', '_', tag)
    tag = tag.strip('_')
    
    # Check for truncated versions first
    for truncated, canonical in TAG_CANONICAL.items():
        if tag == truncated or tag.startswith(truncated) or truncated.startswith(tag):
            return canonical
    
    # Then check for exact matches
    if tag in TAG_CANONICAL:
        return TAG_CANONICAL[tag]
    
    return tag

def normalize_tag_list(tags, category):
    """
    Normalize a list of tags by:
    1. Converting to lowercase
    2. Replacing spaces and special characters with underscores
    3. Removing trailing 's' for singular form
    4. Removing duplicates
    5. Sorting alphabetically
    """
    changes = {
        'singularized': [],
        'cleaned': [],
        'removed': []
    }
    
    normalized = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            continue
            
        original = tag
        # First clean the tag
        tag = clean_tag_name(tag)
        
        # Remove trailing 's' for singular form
        if tag.endswith('s') and len(tag) > 1:
            singular = tag[:-1]
            if singular not in normalized:
                tag = singular
                changes['singularized'].append((original, tag))
        
        # Track changes
        if tag != original:
            changes['cleaned'].append((original, tag))
        
        # Add to normalized list if not empty and not in banlist
        if tag and tag not in normalized and tag not in MATERIAL_BANLIST:
            normalized.append(tag)
        elif tag in MATERIAL_BANLIST:
            changes['removed'].append(tag)
    
    # Sort alphabetically
    normalized.sort()
    
    return normalized, changes

# Set of generic or unhelpful material tags to filter out
MATERIAL_BANLIST = {
    "object", "thing", "item", "element", "structure", "stuff",
    "image", "images", "photo", "view", "display", "visual", "snapshot",
    "scene", "composition", "background", "foreground", "picture", "visuals",
    "eczema_diagnosis", "color_tones"
}

def decide_category(tag, categories):
    """
    Decide which category a tag should belong to based on its semantic meaning.
    Returns the chosen category.
    """
    # Define category priorities
    category_priority = {
        'semantic_tags': 1,  # Highest priority
        'formal_tags': 2,
        'emotional_tags': 3,
        'material_tags': 4   # Lowest priority
    }
    
    # Choose the category with highest priority
    return min(categories, key=lambda x: category_priority[x])

def main():
    print(f"Starting tag normalization in {OUTPUT_DIR}")
    
    # First, generate inconsistency report
    tag_frequency, tag_variants, tag_categories = report_tag_inconsistencies()
    
    # Track all changes made
    all_changes = {
        'singularized': [],
        'cleaned': [],
        'removed': [],
        'category_changes': []
    }
    
    # Process each JSON file
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.json') and not filename.startswith('._'):
            print(f"\nProcessing {filename}...")
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('metadata_payload', {})
                
                # Process each category
                for category in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
                    if category in metadata:
                        print(f"  Normalizing {category}...")
                        original_tags = metadata[category]
                        normalized_tags, changes = normalize_tag_list(original_tags, category)
                        
                        # Track changes
                        all_changes['singularized'].extend(changes['singularized'])
                        all_changes['cleaned'].extend(changes['cleaned'])
                        all_changes['removed'].extend(changes['removed'])
                        
                        # Update the tags
                        metadata[category] = normalized_tags
                        
                        # Handle tags that appear in multiple categories
                        for tag in normalized_tags:
                            if tag in tag_categories and len(tag_categories[tag]) > 1:
                                chosen_category = decide_category(tag, tag_categories[tag])
                                if chosen_category != category:
                                    all_changes['category_changes'].append((tag, category, chosen_category))
                                    if tag in metadata[category]:
                                        metadata[category].remove(tag)
                                    if chosen_category in metadata:
                                        metadata[chosen_category].append(tag)
            
            # Save the updated file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
    
    # Print summary of changes
    print("\nNormalization Summary:")
    print("--------------------")
    
    if all_changes['singularized']:
        print("\nSingularized Tags:")
        for original, normalized in all_changes['singularized']:
            print(f"  {original} -> {normalized}")
    
    if all_changes['cleaned']:
        print("\nCleaned Tags:")
        for original, cleaned in all_changes['cleaned']:
            print(f"  {original} -> {cleaned}")
    
    if all_changes['removed']:
        print("\nRemoved Tags:")
        for tag in all_changes['removed']:
            print(f"  {tag}")
    
    if all_changes['category_changes']:
        print("\nCategory Changes:")
        for tag, old_cat, new_cat in all_changes['category_changes']:
            print(f"  {tag}: {old_cat} -> {new_cat}")
    
    print("\nTag normalization complete!")

if __name__ == "__main__":
    main()