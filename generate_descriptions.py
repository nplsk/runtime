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
INPUT_DIR = "/Volumes/RUNTIME/PROCESSED"
OUTPUT_DIR = INPUT_DIR + "_DESCRIPTIONS"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load once at the top of your script
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
blip_model.eval()



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

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_poetic_metadata(prompt, shared_tags=None):
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

    metadata = data.get("metadata_payload", {})
    semantic_caption = metadata.get("semantic_caption", "")
    motion = metadata.get("motion_score", 0)
    motion_variance = metadata.get("motion_variance", 0)
    dominant_colors = metadata.get("dominant_colors", []) 
    motion_category = metadata.get("motion_tag", "")


    # Interpret dominant color mood
    def estimate_color_mood(colors):
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

    common_tags = get_common_tags()
    shared_terms = list(set(common_tags["semantic_tags"] + common_tags["material_tags"]))

    try:
        ai_outputs = generate_poetic_metadata(prompt, shared_tags=shared_terms)

        # Flatten perceptual_ambient outputs into the top level
        perceptual = ai_outputs.get("perceptual_ambient", {})
        for key, value in perceptual.items():
            data[key] = value

        data["video_path"] = data.get("file_path", "")
        data.pop("metadata_payload", None)

        def normalize_tag_list(tag_list):
            return list(sorted(set(t.strip().lower().replace(" ", "_") for t in tag_list if isinstance(t, str))))

        data["semantic_tags"] = normalize_tag_list(perceptual.get("semantic_tags", []))
        data["formal_tags"] = normalize_tag_list(perceptual.get("formal_tags", []))
        data["emotional_tags"] = normalize_tag_list(perceptual.get("emotional_tags", []))
        data["material_tags"] = normalize_tag_list(perceptual.get("material_tags", []))

        def derive_mood_tag(dominant_colors, motion_tag, emotional_tags):
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

            if total >= 2:
                return "energized"
            elif total == 1:
                return "engaged"
            elif total == 0:
                return "neutral"
            elif total == -1:
                return "subdued"
            else:
                return "tranquil"

        mood_tag = derive_mood_tag(
            dominant_colors=data.get("dominant_colors", []),
            motion_tag=data.get("motion_tag", ""),
            emotional_tags=data.get("emotional_tags", [])
        )
        data["mood_tag"] = mood_tag

        # === Add theme_anchors based on intersection with semantic tags ===
        THEME_KEYWORDS = {
            "family", "forest", "urban", "portrait", "blur", "snow", "serene", "interior",
            "audience", "trees", "sky", "water", "movement", "celebration", "warmth", "cold"
        }
        data["theme_anchors"] = sorted(set(data["semantic_tags"]) & THEME_KEYWORDS)

        # === Add suggested_phase based on semantic tags (parallel to phase_suggestion) ===
        semantic_set = set(data["semantic_tags"])
        if semantic_set & {"forest", "water", "snow", "serene", "trees", "nature", "fog", "sky"}:
            data["suggested_phase"] = "elemental"
        elif semantic_set & {"interior", "architecture", "room", "kitchen", "urban"}:
            data["suggested_phase"] = "built"
        elif semantic_set & {"group", "portrait", "family", "crowd", "musicians", "celebration"}:
            data["suggested_phase"] = "people"
        elif data.get("motion_tag") == "dynamic" or data.get("motion_score", 0) > 30:
            data["suggested_phase"] = "blur"
        else:
            data["suggested_phase"] = "orientation"

        # Insert motion and mood info at top level
        data["motion_score"] = motion
        data["motion_variance"] = motion_variance
        data["motion_tag"] = motion_category
        data["mood_tag"] = color_mood

        # Avoid saving redundant dominant_colors if already present and populated
        if "dominant_colors" not in data or not data["dominant_colors"]:
            data["dominant_colors"] = dominant_colors

        # Update global tag memory with deduplication and sorting
        for k in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
            TAG_MEMORY[k] = sorted(set(TAG_MEMORY.get(k, []) + data.get(k, [])))
        for k in ["motion_tag", "mood_tag"]:
            value = data.get(k)
            if value:
                TAG_MEMORY[k] = sorted(set(TAG_MEMORY.get(k, []) + [value]))
        with open(TAG_HISTORY_PATH, "w") as tf:
            json.dump(TAG_MEMORY, tf)

        # Save the updated JSON
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(json_path))
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Log the generated description and metadata
        with open("descriptions_log.txt", "a") as log_file:
            log_file.write(f"{data['video_path']}:\n")
            log_file.write(f"Description: {data['ai_description']}\n")
            log_file.write(f"Formal Tags: {data['formal_tags']}\n")
            log_file.write(f"Emotional Tags: {data['emotional_tags']}\n")
            log_file.write(f"Semantic Tags: {data['semantic_tags']}\n")
            log_file.write(f"Dominant Colors: {data['dominant_colors']}\n")
            log_file.write("-" * 80 + "\n\n")  # Separator between videos

        print(f"Updated: {output_path}")

    except Exception as e:
        print(f"Error updating {json_path}: {e}")

# === Run it ===
from collections import Counter, defaultdict
import argparse

def recount_tags(output_dir):
    tag_counter = defaultdict(Counter)

    for filename in os.listdir(output_dir):
        if not filename.endswith(".json") or filename.startswith("._"):
            continue
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
                tag_counter[k].update(data.get(k, []))
            for k in ["motion_tag", "mood_tag"]:
                v = data.get(k)
                if v:
                    tag_counter[k][v] += 1
        except Exception as e:
            print(f"Failed counting tags in {filename}: {e}")

    with open("tag_history_lengths.json", "w") as out_f:
        json.dump({k: dict(v) for k, v in tag_counter.items()}, out_f, indent=2)

    print("✅ Tag counts saved to tag_history_lengths.json")

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