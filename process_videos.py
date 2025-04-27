import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"  # ensure it knows ffmpeg exists
os.environ["IMAGEIO_FFMPEG_LOGLEVEL"] = "error"  # suppress verbose output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Optional flag to allow re-enabling tokenizer parallelism if needed
DISABLE_TOKENIZER_PARALLELISM = True
if DISABLE_TOKENIZER_PARALLELISM:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
import cv2
import json
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from moviepy import *
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import sys
import torchvision.transforms as transforms
import re
import nltk
from nltk.stem import WordNetLemmatizer
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

BAD_PHRASES = [
    "i like", "you", "we", "i think", "if you are working", "my project", "looks like", "it looks like", "feel like", "imagine", "screenshot"
]

BAD_START_PHRASES = [
    "if ", "it looks like", "it appears", "there is a", "this is", "this appears", "might be", "could be"
]

# Initialize models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
VIDEO_DIR = "./scenes"
OUTPUT_DIR = "./output"
THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
SEGMENT_DURATION = 5  # seconds

# Supported video formats
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", 
    device_map="auto", 
    torch_dtype=torch.float16
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def calculate_motion_score(frames):
    diffs = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        motion_score = np.mean(diff)
        diffs.append(motion_score)
    return np.mean(diffs) if diffs else 0


PROMPT_KEYWORDS = [
    "describe the setting", "describe the lighting", "identify the time of day", "describe the overall visual impression", "list the main objects"
]

def is_valid_caption(caption):
    """Check if a generated caption is valid."""
    lowered = caption.lower().strip()
    if any(bad_phrase in lowered for bad_phrase in BAD_PHRASES):
        return False
    if any(keyword in lowered for keyword in PROMPT_KEYWORDS):
        return False
    if any(lowered.startswith(bad_start) for bad_start in BAD_START_PHRASES):
        return False
    if len(lowered.split()) < 4:  # Less than 4 words = likely junk
        return False
    return True

def is_duplicate_caption(new_caption, existing_captions, similarity_threshold=0.90):
    """Check if a new caption is too similar to any existing ones."""
    from difflib import SequenceMatcher
    for existing_caption in existing_captions:
        ratio = SequenceMatcher(None, new_caption.lower(), existing_caption.lower()).ratio()
        if ratio > similarity_threshold:
            return True
    return False

def contains_unwanted_place(caption):
    """Check if the caption hallucinated a famous place name."""
    unwanted_places = ["london", "argentina", "paris", "tokyo", "berlin", "san francisco", "chicago", "germany", "california", "texas", "florida", "new jersey", "new mexico", "new hampshire", "new york", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming"]
    lowered = caption.lower()
    return any(place in lowered for place in unwanted_places)

# Add capture time and default location
def get_creation_time(path):
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).isoformat()

def extract_dominant_colors(frame, k=3):
    pixels = frame.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in colors]

def process_video(video_path):
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    logger.info(f"Processing: {video_filename}")

    clip = VideoFileClip(video_path)
    duration = clip.duration
    resolution = clip.size
    frame_rate = clip.fps

    segments = []
    segment_thumbnails = []

    try:
        # Capture thumbnails across the entire video (multi-frame context)
        for start in np.arange(0, duration, SEGMENT_DURATION):
            end = min(start + SEGMENT_DURATION, duration)
            segment_frames = []

            for t in np.linspace(start, end, num=5):
                frame = clip.get_frame(t)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                segment_frames.append(gray)

                # Save thumbnails
                thumb_img = Image.fromarray(frame)
                thumb_path = os.path.join(THUMBNAIL_DIR, f"{video_id}_{int(start)}.jpg")
                thumb_img.save(thumb_path)
                segment_thumbnails.append(thumb_img)

            motion_score = calculate_motion_score(segment_frames)

            segments.append({
                "start": float(start),
                "end": float(end),
                "motion_score": float(round(motion_score, 3)),
                "thumbnail": thumb_path
            })

        # === MULTI-FRAME CONTEXT AVERAGING ===
        # Commented out the composite image generation block
        # if segment_thumbnails:
        #     composite = np.mean([np.array(img) for img in segment_thumbnails], axis=0).astype(np.uint8)
        #     composite_image = Image.fromarray(composite).convert("RGB")
        # else:
        #     logger.warning(f"No thumbnails generated for {video_filename}. Using black frame.")
        #     composite_image = Image.new("RGB", (resolution[0], resolution[1]), color=(0, 0, 0))

        # === CAPTION GENERATION ===
        if segment_thumbnails:
            middle_index = len(segment_thumbnails) // 2
            selected_thumbnail = segment_thumbnails[middle_index]
        else:
            selected_thumbnail = None

        # === Non-prompted (zero-shot) caption ===
        try:
            inputs_raw = blip_processor(images=selected_thumbnail, return_tensors="pt").to(blip_model.device, torch.float16)
            out_raw = blip_model.generate(
                **inputs_raw,
                do_sample=True,
                top_p=0.95,
                temperature=0.3,
                max_new_tokens=60,
                repetition_penalty=1.2,
                num_beams=5,
                min_length=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            raw_caption = blip_processor.tokenizer.decode(out_raw[0], skip_special_tokens=True)
            logger.info(f"Raw caption (no prompt): {raw_caption}")
        except Exception as e:
            logger.warning(f"Error generating raw caption: {e}")
            raw_caption = "No raw caption generated"

        prompts = [
            "What is the primary subject of this scene?",
            "Determine the time of day in this scene",
            "What are the objects visible in the scene?",          
            "What are the actions occurring in the scene?",                    
        ]

        all_captions = []

        if selected_thumbnail:
            for prompt in prompts:
                try:
                    inputs = blip_processor(
                        images=selected_thumbnail,
                        text=prompt,
                        return_tensors="pt"
                    ).to(blip_model.device, torch.float16)

                    out = blip_model.generate(
                        **inputs,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.3,
                        max_new_tokens=60,
                        repetition_penalty=1.2,
                        num_beams=5,
                        min_length=10,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )

                    caption = blip_processor.tokenizer.decode(out[0], skip_special_tokens=True)

                    if caption and len(caption.strip()) > 0:
                        if (
                            is_valid_caption(caption) and
                            not contains_unwanted_place(caption) and
                            not is_duplicate_caption(caption, all_captions, similarity_threshold=0.92)
                        ):
                            all_captions.append(caption.strip())
                            logger.info(f"Accepted caption: {caption.strip()}")
                            if len(all_captions) >= 5:
                                break
                        else:
                            logger.warning(f"Rejected caption (invalid/duplicate/place): {caption.strip()}")
                            with open("rejected_captions_log.txt", "a") as reject_log:
                                reject_log.write(f"{video_filename} - Rejected Caption: {caption.strip()} (Prompt: {prompt})\n")
                    else:
                        logger.warning(f"Empty caption generated for prompt: {prompt}")

                except Exception as e:
                    logger.warning(f"Error generating caption for prompt '{prompt}': {e}")

        # === FINALIZE DATA ===
        if all_captions:
            semantic_caption = " | ".join([raw_caption] + all_captions)
        else:
            semantic_caption = raw_caption if raw_caption else "No caption could be generated"

        # Trim semantic_caption to ~50 words max
        if len(semantic_caption.split()) > 50:
            words = semantic_caption.split()
            semantic_caption = " ".join(words[:50]) + "..."

        dominant_colors = extract_dominant_colors(np.array(segment_thumbnails[0]) if segment_thumbnails else np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))
        embedding = embed_model.encode(semantic_caption, convert_to_tensor=True).cpu().numpy().tolist()

        data = {
            "video_id": video_id,
            "file_path": video_path,
            "duration": duration,
            "semantic_tags": [],
            "semantic_caption": semantic_caption,
            "ai_description": "",
            "formal_tags": [],
            "emotional_tags": [],
            "resolution": resolution,
            "frame_rate": frame_rate,
            "dominant_colors": dominant_colors,
            "creation_time": get_creation_time(video_path),
            "segments": segments,
            "semantic_embedding": embedding,
            "preliminary_tags": []  # (We can auto-generate these later if you want!)
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"{video_id}.json"), "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Finished processing: {video_filename}")

    except Exception as e:
        logger.error(f"Critical error while processing {video_filename}: {e}")

# === Run Processing ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.isfile(video_path):
            process_video(video_path)
        else:
            logger.error(f"File not found: {video_path}")
    else:
        for filename in os.listdir(VIDEO_DIR):
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                process_video(os.path.join(VIDEO_DIR, filename))