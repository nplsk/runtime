"""
This script validates, filters, and regenerates video captions for quality control.
It provides a complete workflow for:
1. Extracting captions from JSON metadata files
2. Identifying problematic captions
3. Regenerating poor quality captions
4. Cleaning up persistently problematic captions
5. Updating the JSON files with improved captions

Input:
- JSON metadata files in data/descriptions/

Output:
- Updated JSON files with improved captions
- all_captions.csv: Current state of all captions
- flagged_weird_captions.csv: Captions that failed validation
- regeneration_attempts.json: Tracking of regeneration attempts
"""

import csv
import json
from pathlib import Path
import sys
import os
import torch
from PIL import Image
from tqdm import tqdm
import re

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import config
from config import MAX_REGENERATION_ATTEMPTS
from process_videos import process_video, blip_processor, blip_model

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Phrases that indicate potential hallucination or incorrect descriptions
BANNED_PHRASES = [
    "airplane", "elephant", "unicorn", "flying in the sky", "close-up of an object",
    "is a screen that is a screen",  # Catch repetitive screen descriptions
    "is a tv that is a tv",  # Catch repetitive TV descriptions
    "is a background that is a background",  # Catch repetitive background descriptions
    "a symphony of",  # Catch symphony patterns
    "a symphony",  # Catch symphony patterns
    "in which the subject",  # Catch generic subject references
    "a symbol of a",  # Catch symbolic references
    "a tv screen background",  # Catch redundant screen/background combinations
    "a tv screen with a tv screen",  # Catch redundant screen references
    "in a sand beach",  # Catch problematic beach descriptions
    "in a light blue",  # Catch problematic color descriptions
    "in a sandstone",  # Catch problematic material descriptions
    "in a fabric",  # Catch problematic fabric descriptions
]
GENERIC_PHRASES = ["a black and white image"]

def generate_new_caption(thumbnail_path):
    """Generate a new caption for a video using BLIP-2."""
    try:
        # Load and process the thumbnail
        image = Image.open(thumbnail_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt").to(blip_model.device)
        
        # Generate caption
        with torch.no_grad():
            outputs = blip_model.generate(
                **inputs,
                do_sample=True,
                top_p=0.9,
                temperature=0.5,
                max_new_tokens=60,
                repetition_penalty=1.2,
                num_beams=7,
                min_length=10,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        caption = blip_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None

def extract_captions_from_json():
    """Extract captions from JSON metadata files and write to CSV."""
    # Create output directory if it doesn't exist
    output_dir = Path("data/descriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open CSV file for writing
    with open("all_captions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "caption"])
        
        # Process each JSON file
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, "r") as jf:
                    data = json.load(jf)
                    
                # Extract video_id and captions
                video_id = data.get("video_id")
                if not video_id:
                    continue
                    
                # Get captions from segments only
                if "segments" in data:
                    for segment in data["segments"]:
                        if "captions" in segment:
                            for caption in segment["captions"]:
                                if caption:  # Only write non-empty captions
                                    writer.writerow([video_id, caption])
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

def update_captions_in_json(json_path, bad_captions, regeneration_attempts):
    """Update only the problematic segment captions in a JSON file."""
    try:
        # Load existing metadata
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        video_id = data.get("video_id")
        if not video_id:
            return False
            
        # Track if we made any changes
        made_changes = False
        
        # Initialize attempt tracking for this video if not exists
        if video_id not in regeneration_attempts:
            regeneration_attempts[video_id] = {"total_attempts": 0, "regenerated_captions": set()}
        
        # Update segment captions only
        if "segments" in data:
            for segment in data["segments"]:
                if "captions" in segment:
                    new_captions = []
                    for caption in segment["captions"]:
                        if caption in bad_captions:
                            # Check if we've exceeded regeneration attempts
                            if regeneration_attempts[video_id]["total_attempts"] >= MAX_REGENERATION_ATTEMPTS:
                                print(f"Removing persistently problematic caption for {video_id}: {caption[:50]}...")
                                continue  # Skip this caption
                            
                            # Check if we've already tried to regenerate this specific caption
                            if caption in regeneration_attempts[video_id]["regenerated_captions"]:
                                print(f"Skipping already regenerated caption for {video_id}: {caption[:50]}...")
                                continue
                            
                            # Generate new caption using the segment's thumbnail
                            new_caption = generate_new_caption(segment["thumbnail_path"])
                            if new_caption:
                                new_captions.append(new_caption)
                                made_changes = True
                                regeneration_attempts[video_id]["total_attempts"] += 1
                                regeneration_attempts[video_id]["regenerated_captions"].add(caption)
                        else:
                            new_captions.append(caption)
                    segment["captions"] = new_captions
        
        # Save changes if we made any
        if made_changes:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {json_path}: {e}")
        return False

def is_repetitive_caption(caption):
    """Check if a caption contains repetitive patterns."""
    # Split into words and check for repeating sequences
    words = caption.lower().split()
    if len(words) < 4:  # Too short to be repetitive
        return False
        
    # Check for repeating 3-word sequences
    for i in range(len(words) - 3):
        sequence = " ".join(words[i:i+3])
        if caption.lower().count(sequence) > 1:
            return True
            
    # Check for repeating phrases
    for phrase in ["screen", "tv", "background", "image", "picture", "sand", "solitary", 
                  "beach", "blue", "stone", "fabric", "light"]:
        if caption.lower().count(phrase) > 2:  # More than 2 occurrences is suspicious
            return True
            
    # Check for repeating single words
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
        if word_counts[word] > 3:  # More than 3 occurrences of any word is suspicious
            return True
            
    # Check for repeating color descriptions
    color_words = ["blue", "red", "green", "yellow", "white", "black", "gray", "grey"]
    for color in color_words:
        if caption.lower().count(color) > 2:
            return True
            
    # Check for repeating material descriptions
    material_words = ["sand", "stone", "fabric", "metal", "wood", "plastic", "glass"]
    for material in material_words:
        if caption.lower().count(material) > 2:
            return True
            
    return False

def is_similar_to_others(caption, other_captions):
    """Check if a caption is too similar to other captions."""
    caption = caption.lower()
    
    # Check for problematic patterns
    problematic_patterns = [
        r"a\s+(\w+)\s+\1",  # Catches "a sand sand sand" - now properly captures the word
        r"a\s+solitary\s+solitary",  # Catches "a solitary solitary"
        r"a\s+tv\s+screen\s+with\s+a\s+tv\s+screen",  # Catches redundant screen references
        r"in\s+a\s+(\w+)\s+\1",  # Catches "in a sand sand"
        r"a\s+(\w+)\s+in\s+a\s+\1",  # Catches "a sand in a sand"
        r"a\s+(\w+)\s+and\s+a\s+\1",  # Catches "a sand and a sand"
    ]
    
    for pattern in problematic_patterns:
        if re.search(pattern, caption):
            return True
    
    # Check for excessive commas
    if caption.count(',') > 3:
        return True
    
    # Check similarity with other captions
    for other in other_captions:
        other = other.lower()
        # If captions share more than 70% of their words, they're too similar
        caption_words = set(caption.split())
        other_words = set(other.split())
        common_words = caption_words.intersection(other_words)
        if len(common_words) / max(len(caption_words), len(other_words)) > 0.7:
            return True
    return False

def regenerate_semantic_captions():
    """Create semantic captions by aggregating the best segment captions."""
    print(f"\n{Colors.BLUE}Creating semantic captions from best segment captions...{Colors.ENDC}")
    
    for json_file in Path(config.OUTPUT_DIR).glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Collect all segment captions
            all_captions = []
            if "segments" in data:
                for segment in data["segments"]:
                    if "captions" in segment:
                        all_captions.extend(segment["captions"])
            
            if all_captions:
                # Filter out problematic captions
                filtered_captions = []
                for caption in all_captions:
                    if not caption:
                        continue
                        
                    # Skip if caption is too short
                    if len(caption) < 10:
                        continue
                        
                    # Skip if caption contains banned phrases
                    if any(phrase in caption.lower() for phrase in BANNED_PHRASES):
                        continue
                        
                    # Skip if caption is repetitive
                    if is_repetitive_caption(caption):
                        continue
                        
                    # Skip if caption is too similar to others we've kept
                    if is_similar_to_others(caption, filtered_captions):
                        continue
                        
                    filtered_captions.append(caption)
                
                # Create semantic caption by joining filtered captions with " | "
                if filtered_captions:
                    semantic_caption = " | ".join(filtered_captions)
                    
                    # Ensure the caption isn't too long
                    if len(semantic_caption) > 500:
                        semantic_caption = semantic_caption[:497] + "..."
                    
                    if "metadata_payload" not in data:
                        data["metadata_payload"] = {}
                    data["metadata_payload"]["semantic_caption"] = semantic_caption
                    
                    # Save changes
                    with open(json_file, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"{Colors.GREEN}Updated semantic caption for {data.get('video_id')}{Colors.ENDC}")
                else:
                    print(f"{Colors.YELLOW}No valid captions found for {data.get('video_id')}{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error updating semantic caption for {json_file}: {e}{Colors.ENDC}")

def cleanup_persistent_bad_captions(regeneration_attempts):
    """Remove only the problematic segment captions that have exceeded regeneration attempts."""
    # Find videos that have exceeded max attempts
    videos_to_cleanup = {
        video_id for video_id, attempts_data in regeneration_attempts.items()
        if attempts_data.get("total_attempts", 0) >= MAX_REGENERATION_ATTEMPTS
    }
    
    if not videos_to_cleanup:
        return
        
    print(f"\n{Colors.YELLOW}Cleaning up persistent bad captions for {len(videos_to_cleanup)} videos...{Colors.ENDC}")
    
    # Process each video's JSON file
    for video_id in tqdm(videos_to_cleanup):
        json_path = Path(config.OUTPUT_DIR) / f"{video_id}.json"
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            made_changes = False
            
            # Get the set of problematic captions for this video
            problematic_captions = set()
            if os.path.exists("flagged_weird_captions.csv"):
                with open("flagged_weird_captions.csv", "r") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if row[0] == video_id:
                            problematic_captions.add(row[1])
            
            # Clean up segment captions only
            if "segments" in data:
                for segment in data["segments"]:
                    if "captions" in segment:
                        # Keep only non-problematic captions
                        original_captions = segment["captions"]
                        segment["captions"] = [c for c in original_captions if c not in problematic_captions]
                        if len(segment["captions"]) != len(original_captions):
                            made_changes = True
            
            # Save changes if we made any
            if made_changes:
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"{Colors.GREEN}Cleaned up problematic captions for {video_id}{Colors.ENDC}")
                
        except Exception as e:
            print(f"{Colors.RED}Error cleaning up {json_path}: {e}{Colors.ENDC}")
    
    # Remove these videos from regeneration attempts
    for video_id in videos_to_cleanup:
        regeneration_attempts.pop(video_id, None)
    
    # Save updated regeneration attempts
    with open("regeneration_attempts.json", "w") as f:
        json_safe_attempts = {}
        for video_id, data in regeneration_attempts.items():
            json_safe_attempts[video_id] = {
                "total_attempts": data["total_attempts"],
                "regenerated_captions": list(data["regenerated_captions"])
            }
        json.dump(json_safe_attempts, f, indent=2)
    
    print(f"\n{Colors.GREEN}Successfully cleaned up problematic captions for {len(videos_to_cleanup)} videos.{Colors.ENDC}")

def process_captions(max_iterations=3):
    """Main workflow for caption validation and regeneration."""
    print(f"{Colors.BLUE}Extracting captions from JSON files...{Colors.ENDC}")
    extract_captions_from_json()
    
    # Load regeneration attempts from file if it exists
    regeneration_attempts = {}
    if os.path.exists("regeneration_attempts.json"):
        try:
            with open("regeneration_attempts.json", "r") as f:
                loaded_attempts = json.load(f)
                # Convert lists back to sets for in-memory use
                for video_id, data in loaded_attempts.items():
                    regeneration_attempts[video_id] = {
                        "total_attempts": data["total_attempts"],
                        "regenerated_captions": set(data["regenerated_captions"])
                    }
        except json.JSONDecodeError:
            print(f"{Colors.YELLOW}Warning: regeneration_attempts.json is malformed. Creating new file.{Colors.ENDC}")
            with open("regeneration_attempts.json", "w") as f:
                json.dump({}, f)
    
    # Check for problematic captions
    print(f"\n{Colors.BLUE}Checking captions for issues...{Colors.ENDC}")
    problematic_captions = []
    
    with open("all_captions.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            video_id, caption = row
            if not caption:
                continue
                
            # Check for various issues
            if len(caption) < 10:  # Too short
                problematic_captions.append([video_id, caption, "Too short"])
            elif any(phrase in caption.lower() for phrase in BANNED_PHRASES):
                problematic_captions.append([video_id, caption, "Contains banned phrase"])
            elif any(phrase in caption.lower() for phrase in GENERIC_PHRASES):
                problematic_captions.append([video_id, caption, "Too generic"])
    
    if problematic_captions:
        # Write problematic captions to CSV
        with open("flagged_weird_captions.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "caption", "reason"])
            writer.writerows(problematic_captions)
        
        print(f"\n{Colors.YELLOW}Found {len(problematic_captions)} problematic captions.{Colors.ENDC}")
        
        # Group bad captions by video_id
        bad_captions_by_video = {}
        for video_id, caption, _ in problematic_captions:
            if video_id not in bad_captions_by_video:
                bad_captions_by_video[video_id] = []
            bad_captions_by_video[video_id].append(caption)
        
        # Process each video's JSON file
        print(f"\n{Colors.BLUE}Regenerating problematic captions...{Colors.ENDC}")
        updated_count = 0
        for video_id, bad_captions in tqdm(bad_captions_by_video.items()):
            json_path = Path(config.OUTPUT_DIR) / f"{video_id}.json"
            if json_path.exists():
                if update_captions_in_json(json_path, bad_captions, regeneration_attempts):
                    updated_count += 1
            else:
                print(f"Could not find JSON file for {video_id}")
        
        print(f"\n{Colors.GREEN}Updated captions in {updated_count} files.{Colors.ENDC}")
        
        # Clean up persistent bad captions
        cleanup_persistent_bad_captions(regeneration_attempts)
        
        # Update all_captions.csv with current state
        print(f"\n{Colors.BLUE}Updating all_captions.csv with current captions...{Colors.ENDC}")
        extract_captions_from_json()
        
        # Run another check if we haven't exceeded max iterations
        if max_iterations > 1:
            print(f"\n{Colors.BLUE}Running final check on captions (attempts remaining: {max_iterations-1})...{Colors.ENDC}")
            process_captions(max_iterations - 1)
        else:
            print(f"\n{Colors.YELLOW}Maximum regeneration attempts reached. Some captions may still be problematic.{Colors.ENDC}")
            print("You may need to manually review and clean up remaining problematic captions.")
            
            # Regenerate semantic captions after all segment captions are processed
            regenerate_semantic_captions()
    else:
        print(f"\n{Colors.GREEN}No problematic captions found. All captions passed validation.{Colors.ENDC}")
        if os.path.exists("flagged_weird_captions.csv"):
            os.remove("flagged_weird_captions.csv")
        
        # Regenerate semantic captions if all segment captions are good
        regenerate_semantic_captions()

def main():
    """Main entry point."""
    process_captions()

if __name__ == "__main__":
    main()