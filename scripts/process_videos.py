"""
This script processes video files to extract metadata, generate descriptions, and analyze visual content.
It uses multiple AI models (BLIP-2, Sentence Transformers) to analyze video content and generate
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
# Aggressive suppression of ffmpeg logs
os.environ["IMAGEIO_NO_INTERNET"] = "1"
os.environ["FFMPEG_BINARY"] = "ffmpeg"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"
os.environ["FFMPEG_SHOW_VERBOSE"] = "0"  # Suppress verbose output from ffmpeg
os.environ["FFPROBE_BINARY"] = "ffprobe"

# Optional flag to allow re-enabling tokenizer parallelism if needed
DISABLE_TOKENIZER_PARALLELISM = True
if DISABLE_TOKENIZER_PARALLELISM:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Filter out specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.video.io.ffmpeg_reader")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.image_processing_utils_fast")
warnings.filterwarnings("ignore", category=UserWarning)  # More aggressive warning suppression

# Suppress ffmpeg and library output
import logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('ffmpeg').setLevel(logging.ERROR)
logging.getLogger('ffprobe').setLevel(logging.ERROR)
logging.getLogger('imageio').setLevel(logging.ERROR)
logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

# Suppress moviepy's verbose output
import sys
import subprocess
from contextlib import contextmanager

# Use DEVNULL for all subprocess calls
DEVNULL = subprocess.DEVNULL

@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Constants
SEGMENT_DURATION = 5.0  # Duration of each segment in seconds

# Default phrases to filter out bad captions
BAD_PHRASES = [
    "i like", "you", "we", "i think", "if you are working", "my project", "looks like", 
    "it looks like", "feel like", "imagine", "screenshot"
]

BAD_START_PHRASES = [
    "if ", "it looks like", "it appears", "there is a", "this is", "this appears", 
    "might be", "could be"
]

# Keywords to guide BLIP-2 caption generation
PROMPT_KEYWORDS = [
    "describe the setting", "describe the lighting", "identify the time of day", 
    "describe the overall visual impression", "list the main objects"
]

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
import torchvision.transforms as transforms
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path
import config
import sys
from contextlib import contextmanager

# Setup logging with color and formatting
class CustomFormatter(logging.Formatter):
    """Custom formatter that adds colors and formatting to log messages."""
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
        'ITALIC': '\033[3m',
        'UNDERLINE': '\033[4m',
        'BLINK': '\033[5m',
        'REVERSE': '\033[7m',
        'HIDDEN': '\033[8m',
        
        # Foreground colors
        'BLACK': '\033[30m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        
        # Background colors
        'BG_BLACK': '\033[40m',
        'BG_RED': '\033[41m',
        'BG_GREEN': '\033[42m',
        'BG_YELLOW': '\033[43m',
        'BG_BLUE': '\033[44m',
        'BG_MAGENTA': '\033[45m',
        'BG_CYAN': '\033[46m',
        'BG_WHITE': '\033[47m'
    }

    # Format for different log levels
    FORMATS = {
        logging.DEBUG: f"{COLORS['DIM']}DEBUG: %(msg)s{COLORS['RESET']}",
        logging.INFO: f"{COLORS['GREEN']}%(msg)s{COLORS['RESET']}",
        logging.WARNING: f"{COLORS['YELLOW']}⚠️  %(msg)s{COLORS['RESET']}",
        logging.ERROR: f"{COLORS['RED']}❌ %(msg)s{COLORS['RESET']}",
        logging.CRITICAL: f"{COLORS['BG_RED']}{COLORS['WHITE']}CRITICAL: %(msg)s{COLORS['RESET']}"
    }

    def format(self, record):
        """Format the log record with appropriate colors and styling."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Apply custom formatter
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Only show INFO and above

# Initialize sentence transformer for semantic analysis
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize BLIP-2 model for image captioning
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", 
    device_map="auto", 
    torch_dtype=torch.float16
)

# Create output directories
os.makedirs(config.THUMBNAILS_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Override MoviePy's VideoFileClip to suppress output
from moviepy.video.io.VideoFileClip import VideoFileClip as OriginalVideoFileClip

class SilentVideoFileClip(OriginalVideoFileClip):
    """A version of VideoFileClip that suppresses all output."""
    
    def __init__(self, filename, *args, **kwargs):
        with suppress_stdout():
            super().__init__(filename, *args, **kwargs)

# Replace VideoFileClip with our silent version
from moviepy import video
video.io.VideoFileClip.VideoFileClip = SilentVideoFileClip

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
    unwanted_places = ["london", "argentina", "paris", "tokyo", "berlin", "san francisco", 
                      "chicago", "germany", "california", "texas", "florida", "new jersey", 
                      "new mexico", "new hampshire", "new york", "ohio", "oklahoma", "oregon", 
                      "pennsylvania", "rhode island", "south carolina", "south dakota", 
                      "tennessee", "texas", "utah", "vermont", "virginia", "washington", 
                      "west virginia", "wisconsin", "wyoming"]
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
        frame: RGB image frame or PIL Image
        k: Number of colors to extract
        
    Returns:
        List of hex color codes for dominant colors
    """
    # Convert PIL Image to numpy array if needed
    if hasattr(frame, 'mode') and not hasattr(frame, 'reshape'):
        import numpy as np
        frame = np.array(frame)
        
    # Ensure we have a 3D RGB array
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (h,w,3), got {frame.shape}")
        
    pixels = frame.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in colors]

# Suppress stdout for subprocess calls
def run_quiet(cmd):
    """Run a command and suppress its output."""
    return subprocess.run(cmd, capture_output=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

def get_video_info(video_path):
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",  # Use quiet instead of error
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name",
        "-show_entries", "format=duration,size,bit_rate", 
        "-print_format", "json",  # Use print_format instead of of
        video_path
    ]
    with suppress_stdout():
        result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
        try:
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed with code {result.returncode}")
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            # If we get invalid JSON, return a minimal dict
            return {
                "streams": [{"width": 0, "height": 0, "r_frame_rate": "0/0", "codec_name": "unknown"}],
                "format": {"duration": "0", "size": "0", "bit_rate": "0"}
            }

def generate_thumbnail(video_path, output_path, size=config.THUMBNAIL_SIZE):
    """Generate a thumbnail from the video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get video duration
    duration_cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-print_format", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    with suppress_stdout():
        result = subprocess.run(duration_cmd, capture_output=True, text=True, stderr=subprocess.DEVNULL)
        try:
            duration = float(result.stdout.strip())
        except (ValueError, AttributeError):
            duration = 0
            
    # Extract frame from 10% into the video
    timestamp = max(0.1, duration * 0.1)
    
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-vf", f"scale={size[0]}:{size[1]}",
        output_path
    ]
    with suppress_stdout():
        subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def normalize_phrase(phrase):
    """Normalize a phrase by removing common artifacts."""
    phrase = phrase.replace(" is shining", " shines")
    phrase = phrase.replace(" is glowing", " glows")
    phrase = phrase.replace(" is appearing", " appears")
    phrase = phrase.replace("scene of ", "")
    phrase = phrase.replace("- & footage", "")
    phrase = phrase.replace("& footage", "")
    phrase = phrase.strip()
    return phrase

def group_similar_phrases_weighted(phrases, threshold=0.80):
    """Group similar phrases using embeddings and cosine similarity."""
    if len(phrases) <= 1:
        return [p[1] for p in phrases]
    embeddings = embed_model.encode([p[1] for p in phrases])
    similarity_matrix = cosine_similarity(embeddings)
    groups = []
    used = set()
    for i, (src_i, phrase_i) in enumerate(phrases):
        if i in used:
            continue
        group = [(src_i, phrase_i)]
        used.add(i)
        for j, (src_j, phrase_j) in enumerate(phrases):
            if j in used:
                continue
            if similarity_matrix[i, j] > threshold:
                group.append((src_j, phrase_j))
                used.add(j)
        group_sorted = sorted(group, key=lambda x: 0 if x[0] == "raw" else 1)
        groups.append(group_sorted[0][1])
    return groups

def classify_phrase(phrase):
    """Classify a phrase into time, mood, or description."""
    lowered = phrase.lower()
    if any(t in lowered for t in ["morning", "afternoon", "evening", "night", "sunset", "sunrise"]):
        return "time"
    if any(m in lowered for m in ["atmosphere", "mood", "feeling", "emotional", "ambience"]):
        return "mood"
    return "description"

def process_video(video_path):
    """
    Process a single video file to extract metadata and generate descriptions.
    Handles video segmentation, thumbnail generation, and metadata extraction.
    
    The process follows these main steps:
    1. Initial video validation and setup
    2. Video segmentation and thumbnail generation
    3. Motion analysis and color extraction
    4. AI-powered caption generation
    5. Metadata compilation and saving
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing processed video metadata
    """
    # === STEP 1: INITIAL SETUP AND VALIDATION ===
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    logger.info(f"🎬 Processing: {video_filename}")

    # Track warnings for this video
    video_warnings = set()

    # Validate video file and get basic properties
    skip_reason = None
    clip = None
    extracted_frames = {}
    middle_frame = None
    
    try:
        # Use a subprocess to get video info without any output
        with suppress_stdout():
            # Load video file and check its properties
            clip = SilentVideoFileClip(video_path)
            video_reader = clip.reader
            try:
                nframes = video_reader.nframes
            except AttributeError:
                # If nframes not available, estimate from duration and fps
                # This is normal for ProRes videos, so we don't add a warning
                nframes = int(video_reader.duration * video_reader.fps)
                
            if clip.duration == 0 or nframes == 0:
                raise ValueError("Unreadable or empty video file.")

        # Store basic video properties
        duration = clip.duration
        resolution = clip.size
        frame_rate = clip.fps
        
        # Extract all frames we'll need up front
        # For each segment
        for start in np.arange(0, duration, SEGMENT_DURATION):
            end = min(start + SEGMENT_DURATION, duration)
            segment_frames = []
            
            # Extract frames from this segment for analysis
            for t in np.linspace(start, end, num=5):
                with suppress_stdout():
                    frame = clip.get_frame(t)
                segment_frames.append(frame)
                
            extracted_frames[start] = segment_frames
            
        # Extract middle frame for later use
        with suppress_stdout():
            middle_frame = clip.get_frame(duration / 2)
        
        # Now that we have all frames, close the clip immediately
        clip.close()
        clip = None
        
    except Exception as e:
        # Handle unreadable videos by moving them to quarantine
        reason = str(e).split('\n')[0].strip()
        if "Error passing `ffmpeg -i` command output" in reason:
            reason = "ffmpeg probe failed – check for corrupt encoding or moov atom issues"
        logger.error(f"Unreadable video file {video_path}: {reason}")
        with open(os.path.join(config.OUTPUT_DIR, "skipped_videos.log"), "a") as skipped_log:
            skipped_log.write(f"{video_path}, reason: {reason}\n")
        try:
            quarantine_dir = os.path.join(os.path.dirname(config.VIDEO_DIR), "QUARANTINE")
            os.makedirs(quarantine_dir, exist_ok=True)
            quarantine_path = os.path.join(quarantine_dir, os.path.basename(video_path))
            os.rename(video_path, quarantine_path)
            logger.warning(f"Moved corrupt file to quarantine: {quarantine_path}")
        except Exception as move_err:
            logger.error(f"Failed to move corrupt file {video_path} to quarantine: {move_err}")
        
        # Make sure to close clip if it was opened
        if clip is not None:
            try:
                clip.close()
            except:
                pass
        return

    try:
        # === STEP 2: VIDEO SEGMENTATION AND THUMBNAIL GENERATION ===
        segments = []
        segment_thumbnails = []

        # Process extracted frames for each segment
        for start, segment_frames in extracted_frames.items():
            end = min(start + SEGMENT_DURATION, duration)
            gray_frames = []
            
            # Process the frames we extracted earlier
            for i, frame in enumerate(segment_frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray_frames.append(gray)

                # Generate and save thumbnail (use the middle frame)
                if i == len(segment_frames) // 2:
                    thumb_img = Image.fromarray(frame)
                    thumb_img = thumb_img.convert("RGB")
                    thumb_img = transforms.functional.center_crop(thumb_img, min(thumb_img.size))
                    thumb_img = thumb_img.resize((364, 364), resample=Image.BICUBIC)
                    thumb_path = os.path.join(config.THUMBNAILS_DIR, f"{video_id}_{int(start)}.jpg")
                    thumb_img.save(thumb_path)
                    segment_thumbnails.append(thumb_img)

            # Calculate motion score for this segment
            motion_score = calculate_motion_score(gray_frames)

            # Store segment information
            segments.append({
                "start": start,
                "end": end,
                "motion_score": motion_score,
                "thumbnail_path": os.path.join(config.THUMBNAILS_DIR, f"{video_id}_{int(start)}.jpg")
            })

        # === STEP 3: AI-POWERED CAPTION GENERATION ===
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

        # === STEP 4: COLOR ANALYSIS ===
        # Extract dominant colors for each segment
        for segment in segments:
            thumb_path = segment["thumbnail_path"]
            thumb_img = cv2.imread(thumb_path)
            thumb_img = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)
            segment["dominant_colors"] = extract_dominant_colors(thumb_img)

        # === STEP 5: KEY FRAME SELECTION ===
        # We already have the middle frame, just need to process the others
        selected_high_motion = max(segments, key=lambda s: s['motion_score'])['thumbnail_path']
        selected_low_motion = min(segments, key=lambda s: s['motion_score'])['thumbnail_path']

        selected_high_motion = Image.open(selected_high_motion).convert("RGB")
        selected_low_motion = Image.open(selected_low_motion).convert("RGB")
        selected_middle = Image.fromarray(middle_frame).convert("RGB")

        selected_frames = [selected_middle, selected_high_motion, selected_low_motion]

        # === STEP 6: FINAL CAPTION GENERATION ===
        all_captions = []
        raw_captions = []
        warnings = []

        # Process each selected frame
        for idx, frame in enumerate(selected_frames):
            if frame is None:
                continue

            # Generate raw caption without specific prompts
            try:
                inputs_raw = blip_processor(images=frame, return_tensors="pt").to(blip_model.device, torch.float16)
                out_raw = blip_model.generate(
                    **inputs_raw,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.5,
                    max_new_tokens=60,
                    repetition_penalty=1.2,
                    num_beams=7,
                    min_length=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                raw_caption = blip_processor.tokenizer.decode(out_raw[0], skip_special_tokens=True)
                if raw_caption:
                    raw_captions.append(("raw", raw_caption.strip()))
            except Exception as e:
                warning_msg = f"Error generating raw caption: {e}"
                video_warnings.add(warning_msg)

            # Generate prompted captions for specific aspects
            prompts = [
                "Describe the composition of the image",
                "Describe the lighting and atmosphere",
                "What time of day is shown?",
                "Describe the overall visual impression"
            ]

            for prompt in prompts:
                try:
                    inputs = blip_processor(
                        images=frame,
                        text=prompt,
                        return_tensors="pt"
                    ).to(blip_model.device, torch.float16)

                    out = blip_model.generate(
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

                    caption = blip_processor.tokenizer.decode(out[0], skip_special_tokens=True)

                    if caption and len(caption.strip()) > 0:
                        if (
                            is_valid_caption(caption) and
                            not contains_unwanted_place(caption) and
                            not is_duplicate_caption(caption, [c[1] for c in all_captions], similarity_threshold=0.95)
                        ):
                            all_captions.append(("prompted", caption.strip()))
                            if len(all_captions) >= 10:
                                break
                    else:
                        warning_msg = f"Empty caption generated for prompt: {prompt}"
                        video_warnings.add(warning_msg)

                except Exception as e:
                    warning_msg = f"Error generating caption for prompt '{prompt}': {e}"
                    video_warnings.add(warning_msg)

        # === STEP 7: FINALIZE METADATA ===
        # Process and organize all generated captions
        accepted_phrases = [
            (src, normalize_phrase(text)) for src, text in raw_captions + all_captions
            if is_valid_caption(text) and not contains_unwanted_place(text) and len(text.split()) >= 4
        ]

        # Group similar phrases and organize by type
        grouped_phrases = group_similar_phrases_weighted(accepted_phrases)
        descriptions = [p for p in grouped_phrases if classify_phrase(p) == "description"]
        times = [p for p in grouped_phrases if classify_phrase(p) == "time"]
        moods = [p for p in grouped_phrases if classify_phrase(p) == "mood"]
        if len(times) > 1:
            times = times[:1]
        final_order = descriptions + times + moods
        semantic_caption = " | ".join(final_order) if final_order else "No valid caption generated"

        # Clean up commercial references
        commercial_junk = ["hd stock video", "royalty free", "stock footage", "high definition footage"]
        for junk in commercial_junk:
            semantic_caption = semantic_caption.replace(junk, "").strip()

        # Calculate motion metrics
        motion_scores_list = [s["motion_score"] for s in segments]
        average_motion_score = float(round(np.mean(motion_scores_list), 3))
        motion_variance = float(round(np.var(motion_scores_list), 3))

        # Determine motion tag based on average score
        if average_motion_score >= 30:
            motion_tag = "dynamic"
        elif average_motion_score >= 15:
            motion_tag = "active"
        elif average_motion_score >= 5:
            motion_tag = "moderate"
        else:
            motion_tag = "still"

        # === STEP 8: COMPILE FINAL METADATA ===
        # Create metadata payload
        metadata_payload = {
            "semantic_caption": semantic_caption.lower(),
            "motion_score": average_motion_score,
            "motion_variance": motion_variance,
            "dominant_colors": extract_dominant_colors(selected_middle),
            "motion_tag": motion_tag
        }

        # Create final data structure
        data = {
            "video_id": video_id,
            "filename": video_filename,
            "duration": duration,
            "resolution": resolution,
            "frame_rate": frame_rate,
            "creation_time": get_creation_time(video_path),
            "segments": segments,
            "metadata_payload": metadata_payload
        }

        # Save to JSON
        output_path = os.path.join(config.OUTPUT_DIR, f"{video_id}.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        # Log any accumulated warnings
        if video_warnings:
            logger.warning(f"📝 Warnings for {video_filename}:")
            for warning in sorted(video_warnings):
                logger.warning(f"  • {warning}")

        logger.info(f"✅ Finished processing: {video_filename}")
        return data

    except Exception as e:
        logger.error(f"❌ Critical error while processing {video_filename}: {e}")
        with open(os.path.join(config.OUTPUT_DIR, "skipped_videos.log"), "a") as skipped_log:
            skipped_log.write(f"{video_path}, reason: CRITICAL ERROR: {e}\n")
        return None

def main():
    """
    Main entry point for video processing.
    Processes all scene video files and generates metadata.
    """
    logger.info("🚀 Starting video processing pipeline")
    
    # Process one video at a time to avoid I/O conflicts
    scene_files = [
        os.path.join(config.SCENES_DIR, filename) 
        for filename in os.listdir(config.SCENES_DIR) 
        if filename.lower().endswith(tuple(ext.lower() for ext in config.VIDEO_EXTENSIONS))
    ]
    
    for video_path in scene_files:
        try:
            process_video(video_path)
        except Exception as exc:
            filename = os.path.basename(video_path)
            logger.error(f"❌ Error processing {filename}: {exc}")
    
    logger.info("✨ Video processing pipeline completed")

if __name__ == "__main__":
    main()