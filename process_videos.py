"""
This script processes video files to extract metadata, generate descriptions, and analyze visual content.
It uses multiple AI models (BLIP-2, CLIP, Sentence Transformers) to analyze video content and generate
rich metadata including:
- Motion analysis
- Color analysis
- Scene descriptions
- Semantic tags
- Emotional and formal characteristics

The script handles video segmentation, thumbnail generation, and metadata extraction for use in
the movement-based video playback system.
"""

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"  # ensure it knows ffmpeg exists
os.environ["IMAGEIO_FFMPEG_LOGLEVEL"] = "error"  # suppress verbose output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Optional flag to allow re-enabling tokenizer parallelism if needed
DISABLE_TOKENIZER_PARALLELISM = True
if DISABLE_TOKENIZER_PARALLELISM:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import required libraries
import cv2
import json
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from moviepy import *
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import sys
import torchvision.transforms as transforms
import re
import nltk
import random
import argparse
import clip
from PIL import Image
import csv

# Initialize CLIP model for visual analysis
model_clip, preprocess_clip = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

# === NLTK Data Check and Preload ===
def ensure_nltk_data():
    """
    Ensure required NLTK data is downloaded and preloaded.
    Downloads WordNet and Open Multilingual WordNet if not present.
    """
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    # Preload WordNet into memory
    from nltk.corpus import wordnet
    _ = wordnet.synsets('test')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Lists of phrases to filter out from generated captions
BAD_PHRASES = [
    "i like", "you", "we", "i think", "if you are working", "my project", "looks like", "it looks like", "feel like", "imagine", "screenshot"
]

BAD_START_PHRASES = [
    "if ", "it looks like", "it appears", "there is a", "this is", "this appears", "might be", "could be"
]

# Initialize sentence transformer for semantic analysis
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
VIDEO_DIR = "/Volumes/RUNTIME/SCENES"
# VIDEO_DIR = "/Volumes/LaCie/CONVERTED/_SOURCE PRO RES"
OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED"
THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
SEGMENT_DURATION = 5  # seconds

# Supported video formats
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')

# Initialize BLIP-2 model for image captioning
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", 
    device_map="auto", 
    torch_dtype=torch.float16
)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def calculate_motion_score(frames):
    """
    Calculate motion intensity score for a sequence of frames.
    Uses frame differencing to measure movement between consecutive frames.
    
    Args:
        frames: List of grayscale frames
        
    Returns:
        Average motion score across all frame pairs
    """
    diffs = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        motion_score = np.mean(diff)
        diffs.append(motion_score)
    return np.mean(diffs) if diffs else 0

# Keywords to guide BLIP-2 caption generation
PROMPT_KEYWORDS = [
    "describe the setting", "describe the lighting", "identify the time of day", "describe the overall visual impression", "list the main objects"
]

def is_valid_caption(caption):
    """
    Check if a generated caption meets quality criteria.
    Filters out low-quality or problematic captions.
    
    Args:
        caption: Generated caption text
        
    Returns:
        Boolean indicating if caption is valid
    """
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
    """
    Check if a new caption is too similar to existing ones.
    Uses sequence matching to detect near-duplicates.
    
    Args:
        new_caption: Caption to check
        existing_captions: List of existing captions
        similarity_threshold: Threshold for considering captions similar
        
    Returns:
        Boolean indicating if caption is a duplicate
    """
    from difflib import SequenceMatcher
    for existing_caption in existing_captions:
        ratio = SequenceMatcher(None, new_caption.lower(), existing_caption.lower()).ratio()
        if ratio > similarity_threshold:
            return True
    return False

def contains_unwanted_place(caption):
    """
    Check if caption contains hallucinated location names.
    Filters out captions that incorrectly identify specific places.
    
    Args:
        caption: Caption text to check
        
    Returns:
        Boolean indicating if caption contains unwanted place names
    """
    unwanted_places = ["london", "argentina", "paris", "tokyo", "berlin", "san francisco", "chicago", "germany", "california", "texas", "florida", "new jersey", "new mexico", "new hampshire", "new york", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming"]
    lowered = caption.lower()
    return any(place in lowered for place in unwanted_places)

def get_creation_time(path):
    """
    Get file creation timestamp.
    
    Args:
        path: Path to file
        
    Returns:
        ISO format timestamp string
    """
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).isoformat()

def extract_dominant_colors(frame, k=3):
    """
    Extract dominant colors from a frame using K-means clustering.
    
    Args:
        frame: RGB image frame
        k: Number of colors to extract
        
    Returns:
        List of hex color codes for dominant colors
    """
    pixels = frame.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in colors]

def process_video(video_path):
    """
    Process a single video file to extract metadata and generate descriptions.
    Handles video segmentation, thumbnail generation, and metadata extraction.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing processed video metadata
    """
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    logger.info(f"Processing: {video_filename}")

    # Check for unreadable or empty video file before processing
    skip_reason = None
    try:
        clip = VideoFileClip(video_path)
        video_reader = clip.reader
        try:
            nframes = video_reader.nframes
        except AttributeError:
            # Estimate frame count if nframes is not available
            nframes = int(video_reader.duration * video_reader.fps)
            skip_reason = "Missing nframes metadata, estimated frame count used"
        if clip.duration == 0 or nframes == 0:
            raise ValueError("Unreadable or empty video file.")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        reason = str(e).split('\n')[0].strip()
        if "Error passing `ffmpeg -i` command output" in reason:
            reason = "ffmpeg probe failed – check for corrupt encoding or moov atom issues"
        logger.error(f"Unreadable video file {video_path}: {e}\n{tb_str}")
        with open(os.path.join(OUTPUT_DIR, "skipped_videos.log"), "a") as skipped_log:
            skipped_log.write(f"{video_path}, reason: {reason}\n")
        try:
            quarantine_dir = os.path.join(os.path.dirname(VIDEO_DIR), "QUARANTINE")
            os.makedirs(quarantine_dir, exist_ok=True)
            quarantine_path = os.path.join(quarantine_dir, os.path.basename(video_path))
            os.rename(video_path, quarantine_path)
            logger.warning(f"Moved corrupt file to quarantine: {quarantine_path}")
        except Exception as move_err:
            logger.error(f"Failed to move corrupt file {video_path} to quarantine: {move_err}")
        return
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
                thumb_img = thumb_img.convert("RGB")
                thumb_img = transforms.functional.center_crop(thumb_img, min(thumb_img.size))
                thumb_img = thumb_img.resize((364, 364), resample=Image.BICUBIC)
                thumb_path = os.path.join(THUMBNAIL_DIR, f"{video_id}_{int(start)}.jpg")
                thumb_img.save(thumb_path)
                segment_thumbnails.append(thumb_img)

            motion_score = calculate_motion_score(segment_frames)

            segments.append({
                "start": start,
                "end": end,
                "motion_score": motion_score,
                "thumbnail_path": os.path.join(THUMBNAIL_DIR, f"{video_id}_{int(start)}.jpg")
            })

        # Generate descriptions for each segment using BLIP-2
        for segment in segments:
            thumb_path = segment["thumbnail_path"]
            thumb_img = Image.open(thumb_path).convert("RGB")
            
            # Generate multiple captions for diversity
            captions = []
            for prompt in PROMPT_KEYWORDS:
                inputs = blip_processor(thumb_img, text=prompt, return_tensors="pt").to(blip_model.device)
                outputs = blip_model.generate(**inputs, max_length=50)
                caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
                
                if is_valid_caption(caption) and not is_duplicate_caption(caption, captions):
                    captions.append(caption)
            
            segment["captions"] = captions

        # Extract dominant colors for each segment
        for segment in segments:
            thumb_path = segment["thumbnail_path"]
            thumb_img = cv2.imread(thumb_path)
            thumb_img = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)
            segment["dominant_colors"] = extract_dominant_colors(thumb_img)

        # Generate semantic tags using CLIP
        for segment in segments:
            thumb_path = segment["thumbnail_path"]
            thumb_img = Image.open(thumb_path).convert("RGB")
            image_input = preprocess_clip(thumb_img).unsqueeze(0).to(model_clip.device)
            
            # Define candidate tags for classification
            candidate_tags = [
                "urban", "nature", "people", "architecture", "abstract", "texture",
                "motion", "still", "bright", "dark", "colorful", "monochrome"
            ]
            
            text_inputs = clip.tokenize(candidate_tags).to(model_clip.device)
            with torch.no_grad():
                image_features = model_clip.encode_image(image_input)
                text_features = model_clip.encode_text(text_inputs)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Select tags with high confidence
                segment["tags"] = [
                    tag for tag, score in zip(candidate_tags, similarity[0])
                    if score > 0.3
                ]

        # Create final metadata structure
        metadata = {
            "video_id": video_id,
            "filename": video_filename,
            "duration": duration,
            "resolution": resolution,
            "frame_rate": frame_rate,
            "creation_time": get_creation_time(video_path),
            "segments": segments
        }

        # Save metadata to JSON file
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully processed {video_filename}")
        return metadata

    except Exception as e:
        logger.error(f"Error processing {video_filename}: {str(e)}")
        return None

def main():
    """
    Main entry point for video processing.
    Processes all video files in the input directory and generates metadata.
    """
    # Ensure NLTK data is available
    ensure_nltk_data()

    # Process all video files in the input directory
    for filename in os.listdir(VIDEO_DIR):
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            video_path = os.path.join(VIDEO_DIR, filename)
            process_video(video_path)

if __name__ == "__main__":
    main()