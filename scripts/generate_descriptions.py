"""
This script generates rich descriptions and metadata for video clips using AI models.
It combines multiple AI systems to analyze video content and generate structured metadata:

1. GPT-4: For creating poetic, ambient descriptions
2. Color Analysis: For extracting dominant colors and mood
3. Motion Analysis: For characterizing movement patterns

The script processes video metadata JSON files and enriches them with:
- AI-generated scene descriptions
- Semantic tags
- Formal tags (composition, lighting, etc.)
- Emotional tags
- Material tags
- Color mood analysis
- Motion characteristics

Key features:
- Parallel processing of multiple files
- Tag normalization and cleaning
- Tag history tracking
- Color mood estimation
- Motion pattern analysis
- Error handling and logging

Environment Variables Required:
- OPENAI_API_KEY: Your OpenAI API key for GPT-4 access
"""

import os
import json
import base64
from openai import OpenAI
from PIL import Image
from datetime import datetime
import torch
import cv2
import numpy as np
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import config
from config import (
    OUTPUT_DIR,
    THUMBNAILS_DIR,
    MAX_CAPTION_LENGTH
)

# Load environment variables
load_dotenv()

# Verify API key is present
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

# Tags to filter out from the final output
GARBAGE_TAGS = [
    "movement_score", "thumbnail_shows", "softly_lit",
    "lighting", "motion_score", "capture_time", "frame", "image"
]

# Global tag memory for tracking tag frequency and patterns
TAG_HISTORY_PATH = "./tag_history.json"
try:
    with open(TAG_HISTORY_PATH, "r") as tf:
        TAG_MEMORY = json.load(tf)
except:
    TAG_MEMORY = {"semantic_tags": [], "material_tags": []}

def get_common_tags():
    """
    Get the most frequently used tags from the tag memory.
    
    Returns:
        Dictionary with most common tags for each category
    """
    counters = {k: Counter(v) for k, v in TAG_MEMORY.items()}
    common = {k: [tag for tag, _ in counters[k].most_common(3)] for k in counters}
    return common

# === CONFIG ===
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def describe_thumbnail(image_path):
    """
    Generate a description and basic tags for a video thumbnail.
    Analyzes brightness and uses BLIP-2 for caption generation.
    
    Args:
        image_path: Path to the thumbnail image
        
    Returns:
        Tuple of (description, basic_tags)
    """
    try:
        # Analyze image brightness
        img = Image.open(image_path).convert("RGB")
        colors = img.getcolors(maxcolors=256000)
        brightness = sum([sum(c[1]) for c in colors]) / (len(colors) * 3)

        # Categorize lighting based on brightness
        if brightness > 180:
            lighting = "bright"
        elif brightness > 100:
            lighting = "softly lit"
        else:
            lighting = "dimly lit"

        # Generate caption using BLIP-2
        inputs = blip_processor(images=img, return_tensors="pt").to("cpu")
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs)
        caption = blip_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract basic tags from caption
        words = re.findall(r'\b\w+\b', caption.lower())
        ignore = {"the", "a", "an", "and", "of", "on", "in", "with", "to", "for", "from", "at", "by", "is", "it", "this"}
        basic_tags = sorted(set(w for w in words if w not in ignore and len(w) > 3))

        return f"A {lighting} frame. The thumbnail shows: {caption}.", basic_tags

    except Exception as e:
        return f"Unable to describe thumbnail: {e}", []

def extract_json_block(text):
    """
    Extract a JSON block from text using regex.
    
    Args:
        text: Text containing a JSON block
        
    Returns:
        Extracted JSON string or None
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def encode_image_to_base64(image_path):
    """
    Convert an image to base64 encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_poetic_metadata(prompt, shared_tags=None):
    """
    Generate poetic metadata using GPT-4.
    Incorporates shared tags from previous footage for continuity.
    
    Args:
        prompt: Base prompt for description
        shared_tags: List of tags from previous footage
        
    Returns:
        Dictionary containing generated metadata
    """
    results = {}
    common_snippet = ""
    if shared_tags:
        common_snippet += f"\n\nEchoes of previous footage: {', '.join(shared_tags)}."

    full_prompt = prompt + "\n\nUse minimal language. Limit ai_description to 300 characters max." + common_snippet

    messages = [{"type": "text", "text": full_prompt}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": messages}]
    )
    ai_output = response.choices[0].message.content
    ai_data_raw = extract_json_block(ai_output)
    try:
        ai_data = json.loads(ai_data_raw)
        results["perceptual_ambient"] = ai_data
    except:
        results["perceptual_ambient"] = {"error": "Failed to parse"}

    return results

def estimate_brightness(image_path):
    """
    Calculate the average brightness of an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Normalized brightness value (0-1)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        colors = img.getcolors(maxcolors=256000)
        brightness = sum([sum(c[1]) for c in colors]) / (len(colors) * 3)
        return brightness / 255.0  # normalize to 0-1
    except:
        return 0.5  # fallback

def clean_tags(tag_list):
    """
    Clean and filter a list of tags.
    Removes garbage tags, short tags, and numbers.
    
    Args:
        tag_list: List of tags to clean
        
    Returns:
        Cleaned and sorted list of unique tags
    """
    cleaned = []
    for tag in tag_list:
        tag = tag.strip().lower()
        if any(g in tag for g in GARBAGE_TAGS):
            continue
        if len(tag) < 3:
            continue
        if tag.isdigit():
            continue
        cleaned.append(tag)
    return list(sorted(set(cleaned)))

def enrich_json(video_path):
    """
    Enrich a video metadata JSON file with AI-generated descriptions and tags.
    
    Args:
        video_path: Path to the video file
    """
    if os.path.basename(video_path).startswith("._"):
        return  # Skip macOS metadata files

    try:
        # Load video metadata
        metadata_path = config.OUTPUT_DIR / f"{video_path.stem}.json"
        if not metadata_path.exists():
            print(f"No metadata found for {video_path}")
            return
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Create metadata_payload from segments data
        metadata_payload = {}
        
        # Collect all captions and data from segments
        all_captions = []
        all_motion_scores = []
        all_dominant_colors = []
        
        if metadata.get("segments"):
            for segment in metadata["segments"]:
                # Collect captions
                if segment.get("captions"):
                    all_captions.extend(segment["captions"])
                
                # Collect motion scores
                if "motion_score" in segment:
                    all_motion_scores.append(segment["motion_score"])
                
                # Collect dominant colors
                if segment.get("dominant_colors"):
                    all_dominant_colors.extend(segment["dominant_colors"])
            
            # Use all unique captions
            metadata_payload["semantic_caption"] = " | ".join(set(all_captions)) if all_captions else ""
            
            # Calculate average motion score
            metadata_payload["motion_score"] = sum(all_motion_scores) / len(all_motion_scores) if all_motion_scores else 0
            # Calculate motion variance from the motion scores
            metadata_payload["motion_variance"] = float(round(np.var(all_motion_scores), 3)) if all_motion_scores else 0
            
            # Get unique dominant colors
            metadata_payload["dominant_colors"] = list(set(all_dominant_colors))
            
            # Derive motion_tag from average motion score
            avg_motion_score = metadata_payload["motion_score"]
            if avg_motion_score > 60:
                metadata_payload["motion_tag"] = "dynamic"
            elif avg_motion_score > 30:
                metadata_payload["motion_tag"] = "moderate"
            else:
                metadata_payload["motion_tag"] = "still"
        else:
            # Default values if no segments found
            metadata_payload = {
                "semantic_caption": "",
                "motion_score": 0,
                "motion_variance": 0,
                "dominant_colors": [],
                "motion_tag": "still"
            }

        def estimate_color_mood(colors):
            """
            Estimate the mood based on dominant colors.
            Analyzes warm vs cool color balance.
            
            Args:
                colors: List of hex color codes
                
            Returns:
                Color mood ("warm", "cool", or "neutral")
            """
            warm = 0
            cool = 0
            for hex_color in colors:
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                if r > b and r > g:
                    warm += 1
                elif b > r and b > g:
                    cool += 1
            if warm > cool:
                return "warm"
            elif cool > warm:
                return "cool"
            else:
                return "neutral"

        color_mood = estimate_color_mood(metadata_payload["dominant_colors"])

        # Construct prompt for GPT-4
        prompt = f"""
Scene Description:
{metadata_payload["semantic_caption"]}

Perceptual Motion: {metadata_payload["motion_tag"]}
Color Mood: {color_mood}

Describe a concise scene. Do not copy any metadata labels or numbers. Describe the spatial layout, dominant motion quality, emotional tone, and lighting conditions of the scene. Avoid storytelling. Do not invent unseen details. Output structured JSON:

{{
  "ai_description": "...",
  "semantic_tags": [...],
  "formal_tags": [...],
  "emotional_tags": [...],
  "material_tags": [...]
}}

"""

        # Get common tags for continuity
        common_tags = get_common_tags()
        shared_terms = list(set(common_tags["semantic_tags"] + common_tags["material_tags"]))

        try:
            # Generate metadata using GPT-4
            ai_outputs = generate_poetic_metadata(prompt, shared_tags=shared_terms)

            # Process and normalize the generated metadata
            perceptual = ai_outputs.get("perceptual_ambient", {})
            for key, value in perceptual.items():
                metadata_payload[key] = value

            metadata_payload["video_path"] = str(video_path)
            metadata_payload.pop("metadata_payload", None)

            def normalize_tag_list(tag_list):
                """Normalize and clean a list of tags."""
                return list(sorted(set(t.strip().lower().replace(" ", "_") for t in tag_list if isinstance(t, str))))

            # Normalize all tag categories
            metadata_payload["semantic_tags"] = normalize_tag_list(perceptual.get("semantic_tags", []))
            metadata_payload["formal_tags"] = normalize_tag_list(perceptual.get("formal_tags", []))
            metadata_payload["emotional_tags"] = normalize_tag_list(perceptual.get("emotional_tags", []))
            metadata_payload["material_tags"] = normalize_tag_list(perceptual.get("material_tags", []))

            def derive_mood_tag(dominant_colors, motion_tag, emotional_tags):
                """
                Derive an overall mood tag based on colors, motion, and emotional tags.
                
                Args:
                    dominant_colors: List of dominant colors
                    motion_tag: Motion category
                    emotional_tags: List of emotional tags
                    
                Returns:
                    Overall mood tag
                """
                warm_keywords = ["warm", "sunset", "gold", "orange", "red"]
                cool_keywords = ["blue", "aqua", "gray", "mist", "fog", "cold"]

                warm_score = sum(1 for c in dominant_colors if any(w in c.lower() for w in warm_keywords))
                cool_score = sum(1 for c in dominant_colors if any(cw in c.lower() for cw in cool_keywords))

                emotional_score = 0
                if "calm" in emotional_tags or "serene" in emotional_tags:
                    emotional_score -= 1
                if "anxious" in emotional_tags or "tense" in emotional_tags:
                    emotional_score += 1

                if motion_tag == "still":
                    emotional_score -= 1
                elif motion_tag == "dynamic":
                    emotional_score += 1

                total = emotional_score + warm_score - cool_score
                if total > 1:
                    return "warm"
                elif total < -1:
                    return "cool"
                else:
                    return "neutral"

            # Add derived mood tag
            metadata_payload["mood_tag"] = derive_mood_tag(
                metadata_payload["dominant_colors"], 
                metadata_payload["motion_tag"], 
                metadata_payload["emotional_tags"]
            )

            # Update metadata
            metadata["metadata_payload"] = metadata_payload
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Enriched metadata for {video_path}")
            return True

        except Exception as e:
            print(f"❌ Error generating metadata for {video_path}: {str(e)}")
            return False

    except Exception as e:
        print(f"❌ Error processing {video_path}: {str(e)}")
        return False

def recount_tags(output_dir):
    """
    Recount and update tag frequencies in the tag history.
    
    Args:
        output_dir: Directory containing enriched JSON files
    """
    # Implementation details...

def generate_description(video_path):
    """Generate AI description for a single video file."""
    try:
        # Load video metadata
        metadata_path = config.OUTPUT_DIR / f"{video_path.stem}.json"
        if not metadata_path.exists():
            print(f"No metadata found for {video_path}")
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Generate description using BLIP-2
        # TODO: Implement BLIP-2 description generation
        
        # Update metadata with description
        metadata['description'] = "Placeholder description"  # Replace with actual BLIP-2 output
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
        
    except Exception as e:
        print(f"Error generating description for {video_path}: {str(e)}")
        return None

def main():
    """Enrich all JSON metadata files in the descriptions directory."""
    json_files = list(config.OUTPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {config.OUTPUT_DIR}")
        return

    print(f"Enriching metadata for {len(json_files)} files...")
    for json_path in tqdm(json_files, desc="Enriching descriptions"):
        enrich_json(json_path)

if __name__ == "__main__":
    main()