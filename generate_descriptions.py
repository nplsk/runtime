"""
This script generates rich descriptions and metadata for video clips using AI models.
It combines multiple AI systems to analyze video content and generate structured metadata:

1. BLIP-2: For generating initial image captions
2. GPT-4: For creating poetic, ambient descriptions
3. Color Analysis: For extracting dominant colors and mood
4. Motion Analysis: For characterizing movement patterns

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
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import cv2
import numpy as np
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

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
INPUT_DIR = "/Volumes/RUNTIME/PROCESSED"
OUTPUT_DIR = INPUT_DIR + "_DESCRIPTIONS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize BLIP-2 model for image captioning
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
blip_model.eval()

# Template for generating poetic, ambient descriptions
perceptual_ambient_prompt = (
    "Describe a short video fragment using minimal language. "
    "Focus on spatial layout, motion, light, and atmosphere. "
    "Avoid metaphor and narrative. Prioritize clarity and tone. "
    "Return structured JSON:\n"
    "{\n"
    '  "ai_description": "...",\n'
    '  "semantic_tags": [...],\n'
    '  "formal_tags": [...],\n'
    '  "emotional_tags": [...],\n'
    '  "material_tags": [...]'
    "\n}"
)

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

def enrich_json(json_path):
    """
    Enrich a video metadata JSON file with AI-generated descriptions and tags.
    
    Args:
        json_path: Path to the input JSON file
    """
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(json_path))
    if os.path.exists(output_path):
        print(f"🔁 Skipping already processed: {output_path}")
        return

    if os.path.basename(json_path).startswith("._"):
        return  # Skip macOS metadata files

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Skipping {json_path}: {e}")
        return

    # Extract existing metadata
    metadata = data.get("metadata_payload", {})
    semantic_caption = metadata.get("semantic_caption", "")
    motion = metadata.get("motion_score", 0)
    motion_variance = metadata.get("motion_variance", 0)
    dominant_colors = metadata.get("dominant_colors", []) 
    motion_category = metadata.get("motion_tag", "")

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

    color_mood = estimate_color_mood(dominant_colors)

    # Construct prompt for GPT-4
    prompt = f"""
Scene Description:
{semantic_caption}

Perceptual Motion: {motion_category}
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
            data[key] = value

        data["video_path"] = data.get("file_path", "")
        data.pop("metadata_payload", None)

        def normalize_tag_list(tag_list):
            """Normalize and clean a list of tags."""
            return list(sorted(set(t.strip().lower().replace(" ", "_") for t in tag_list if isinstance(t, str))))

        # Normalize all tag categories
        data["semantic_tags"] = normalize_tag_list(perceptual.get("semantic_tags", []))
        data["formal_tags"] = normalize_tag_list(perceptual.get("formal_tags", []))
        data["emotional_tags"] = normalize_tag_list(perceptual.get("emotional_tags", []))
        data["material_tags"] = normalize_tag_list(perceptual.get("material_tags", []))

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
        data["mood_tag"] = derive_mood_tag(dominant_colors, motion_category, data["emotional_tags"])

        # Save enriched data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Enriched: {output_path}")

    except Exception as e:
        print(f"❌ Failed to enrich {json_path}: {e}")

def recount_tags(output_dir):
    """
    Recount and update tag frequencies in the tag history.
    
    Args:
        output_dir: Directory containing enriched JSON files
    """
    # Implementation details...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recount_only", action="store_true", help="Only recount tags without regenerating descriptions")
    args = parser.parse_args()

    if args.recount_only:
        recount_tags(OUTPUT_DIR)
    else:
        json_files = [
            os.path.join(INPUT_DIR, filename)
            for filename in os.listdir(INPUT_DIR)
            if filename.endswith(".json")
        ]

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(enrich_json, path) for path in json_files]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in file {json_files[i-1]}: {e}")

        recount_tags(OUTPUT_DIR)