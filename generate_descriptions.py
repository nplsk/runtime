import os
import json
from openai import OpenAI
from PIL import Image
from datetime import datetime
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import cv2
import numpy as np
import re
from collections import Counter

GARBAGE_TAGS = [
    "movement_score", "thumbnail_shows", "softly_lit",
    "lighting", "motion_score", "capture_time", "frame", "image"
]

# Global tag memory
TAG_HISTORY_PATH = "./tag_history.json"
try:
    with open(TAG_HISTORY_PATH, "r") as tf:
        TAG_MEMORY = json.load(tf)
except:
    TAG_MEMORY = {"semantic_tags": [], "material_tags": []}

def get_common_tags():
    counters = {k: Counter(v) for k, v in TAG_MEMORY.items()}
    common = {k: [tag for tag, _ in counters[k].most_common(3)] for k in counters}
    return common

# === CONFIG ===
client = OpenAI(api_key="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")
INPUT_DIR = "./output"

# Load once at the top of your script
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
blip_model.eval()

# Optional: define location metadata manually or automate later
# DEFAULT_LOCATION = {
#     "lat": 0.0,
#     "lon": 0.0,
#     "place": "Unknown Location"
# }

perceptual_ambient_prompt = (
    "Describe a short video fragment using minimal language. "
    "Focus on spatial layout, motion, light, and atmosphere. "
    "Avoid metaphor and narrative. Prioritize clarity and tone. "
    "Return only structured JSON:\n"
    "{\n"
    '  "ai_description": "...",\n'
    '  "semantic_tags": [...],\n'
    '  "formal_tags": [...],\n'
    '  "emotional_tags": [...],\n'
    '  "material_tags": [...]\n'
    "}"
)

def get_creation_time(file_path):
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).isoformat()
    except Exception:
        return "Unknown"

def describe_thumbnail(image_path):
    try:
        # --- Get brightness rating ---
        img = Image.open(image_path).convert("RGB")
        colors = img.getcolors(maxcolors=256000)
        brightness = sum([sum(c[1]) for c in colors]) / (len(colors) * 3)

        if brightness > 180:
            lighting = "bright"
        elif brightness > 100:
            lighting = "softly lit"
        else:
            lighting = "dimly lit"

        # --- Generate BLIP caption ---
        inputs = blip_processor(images=img, return_tensors="pt").to("cpu")
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs)
        caption = blip_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Generate basic tags from caption using simple keyword filtering
        words = re.findall(r'\b\w+\b', caption.lower())
        ignore = {"the", "a", "an", "and", "of", "on", "in", "with", "to", "for", "from", "at", "by", "is", "it", "this"}
        basic_tags = sorted(set(w for w in words if w not in ignore and len(w) > 3))

        return f"A {lighting} frame. The thumbnail shows: {caption}.", basic_tags

    except Exception as e:
        return f"Unable to describe thumbnail: {e}", []

def extract_json_block(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def generate_poetic_metadata(prompt, shared_tags=None):
    results = {}
    common_snippet = ""
    if shared_tags:
        common_snippet = f"\n\nConsider echoes of prior footage containing: {', '.join(shared_tags)}."

    full_prompt = prompt + common_snippet

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": perceptual_ambient_prompt},
            {"role": "user", "content": full_prompt}
        ]
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
    try:
        img = Image.open(image_path).convert("RGB")
        colors = img.getcolors(maxcolors=256000)
        brightness = sum([sum(c[1]) for c in colors]) / (len(colors) * 3)
        return brightness / 255.0  # normalize to 0-1
    except:
        return 0.5  # fallback

def clean_tags(tag_list):
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
    with open(json_path, 'r') as f:
        data = json.load(f)

    video_path = data.get("file_path", "")
    capture_time = get_creation_time(video_path)
    # location = data.get("location", DEFAULT_LOCATION)

    # Handle both segmented and non-segmented videos
    if data.get("segments"):
        # Use the segment with highest motion score
        segment = max(data["segments"], key=lambda s: s.get("motion_score", 0))
        thumb_path = segment.get("thumbnail", "N/A")
        motion = segment.get("motion_score", 0)
    else:
        # For older files without segments, use the main thumbnail
        thumb_path = os.path.join(INPUT_DIR, "thumbnails", f"{os.path.splitext(os.path.basename(json_path))[0]}.jpg")
        if not os.path.exists(thumb_path):
            print(f"Skipping {json_path} — no thumbnail found.")
            return
        motion = 0  # Default motion score for older files

    visual_prompt, basic_tags = describe_thumbnail(thumb_path)

    brightness = estimate_brightness(thumb_path)
    tone_hint = ""
    if motion < 10 and brightness < 0.3:
        tone_hint = "This moment feels hushed, suspended, or hidden in shadow."
    elif motion > 40 and brightness > 0.7:
        tone_hint = "This moment may feel vivid, kinetic, or exposed to intense light."

    common_tags = get_common_tags()
    shared_terms = list(set(common_tags["semantic_tags"] + common_tags["material_tags"]))

    prompt = f"""
Video Metadata (for context only):
```
- Time: {capture_time}
- Motion Score: {motion}
- Thumbnail: {visual_prompt}
- Tone Hint: {tone_hint}
```
Describe a concise scene. Do not copy any metadata labels or numbers.
"""

    try:
        ai_outputs = generate_poetic_metadata(prompt, shared_tags=shared_terms)
        
        # Raw output saving disabled
        # with open(json_path.replace('.json', '_raw.txt'), 'w') as raw_f:
        #     raw_f.write(f"{json.dumps(ai_outputs['perceptual_ambient'], indent=2)}\n")

        # Flatten perceptual_ambient outputs into the top level
        perceptual = ai_outputs.get("perceptual_ambient", {})
        for key, value in perceptual.items():
            data[key] = value

        data["video_path"] = video_path
        data["ai_description"] = perceptual.get("ai_description", "")
        data["preliminary_tags"] = basic_tags
        data["semantic_tags"] = clean_tags(perceptual.get("semantic_tags", []))
        data["formal_tags"] = clean_tags(perceptual.get("formal_tags", []))
        data["emotional_tags"] = clean_tags(perceptual.get("emotional_tags", []))
        data["material_tags"] = clean_tags(perceptual.get("material_tags", []))
        data["dominant_colors"] = data.get("dominant_colors", [])
        data["capture_time"] = capture_time
        # data["location"] = location

        # Update global tag memory
        for k in ["semantic_tags", "material_tags"]:
            TAG_MEMORY[k].extend(data.get(k, []))
        with open(TAG_HISTORY_PATH, "w") as tf:
            json.dump(TAG_MEMORY, tf)

        # Save the updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Log the generated description and metadata
        with open("descriptions_log.txt", "a") as log_file:
            log_file.write(f"{video_path}:\n")
            log_file.write(f"Description: {data['ai_description']}\n")
            log_file.write(f"Formal Tags: {data['formal_tags']}\n")
            log_file.write(f"Emotional Tags: {data['emotional_tags']}\n")
            log_file.write(f"Semantic Tags: {data['semantic_tags']}\n")
            log_file.write(f"Dominant Colors: {data['dominant_colors']}\n")
            log_file.write("-" * 80 + "\n\n")  # Separator between videos
            
        print(f"Updated: {json_path}")

    except Exception as e:
        print(f"Error updating {json_path}: {e}")

# === Run it ===
if __name__ == "__main__":
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            enrich_json(os.path.join(INPUT_DIR, filename))