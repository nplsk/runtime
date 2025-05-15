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
model_clip, preprocess_clip = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

# === NLTK Data Check and Preload ===
def ensure_nltk_data():
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
VIDEO_DIR = "/Volumes/RUNTIME/SCENES"
# VIDEO_DIR = "/Volumes/LaCie/CONVERTED/_SOURCE PRO RES"
OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED"
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
                "start": float(start),
                "end": float(end),
                "motion_score": float(round(motion_score, 3)),
                "thumbnail": thumb_path
            })

        # === MULTI-FRAME CONTEXT SELECTION ===
        if segment_thumbnails:
            middle_index = len(segment_thumbnails) // 2
            selected_middle = segment_thumbnails[middle_index]

            # Calculate motion scores per frame
            motion_scores = [np.mean(cv2.absdiff(np.array(segment_thumbnails[i]), np.array(segment_thumbnails[i-1]))) for i in range(1, len(segment_thumbnails))]
            if motion_scores:
                max_motion_index = np.argmax(motion_scores)
                min_motion_index = np.argmin(motion_scores)
                selected_high_motion = segment_thumbnails[max_motion_index]
                selected_low_motion = segment_thumbnails[min_motion_index]
            else:
                selected_high_motion = selected_middle  # fallback
                selected_low_motion = selected_middle  # fallback
        else:
            selected_middle = None
            selected_high_motion = None
            selected_low_motion = None

        # === CAPTION GENERATION (MULTI-FRAME) ===
        selected_frames = [selected_middle, selected_high_motion, selected_low_motion]
        all_captions = []
        raw_captions = []

        for idx, frame in enumerate(selected_frames):
            if frame is None:
                continue

            # Prepare thumbnail for BLIP
            thumb_img = frame
            # === Raw caption ===
            try:
                inputs_raw = blip_processor(images=thumb_img, return_tensors="pt").to(blip_model.device, torch.float16)
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
                if raw_caption:
                    raw_captions.append(("raw", raw_caption.strip()))
                    logger.info(f"Raw caption (frame {idx}): {raw_caption.strip()}")
            except Exception as e:
                logger.warning(f"Error generating raw caption (frame {idx}): {e}")

            # === Prompted captions ===
            prompts = [
                "Describe the composition of the image"
            ]

            for prompt in prompts:
                try:
                    inputs = blip_processor(
                        images=thumb_img,
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
                            not is_duplicate_caption(caption, [c[1] for c in all_captions], similarity_threshold=0.92)
                        ):
                            all_captions.append(("prompted", caption.strip()))
                            logger.info(f"Prompted caption (frame {idx}, prompt '{prompt}'): {caption.strip()}")
                            if len(all_captions) >= 10:
                                break
                    else:
                        logger.warning(f"Empty caption generated for prompt: {prompt}")

                except Exception as e:
                    logger.warning(f"Error generating caption for prompt '{prompt}': {e}")

        # === FINALIZE semantic_caption ===
        def normalize_phrase(phrase):
            phrase = phrase.replace(" is shining", " shines")
            phrase = phrase.replace(" is glowing", " glows")
            phrase = phrase.replace(" is appearing", " appears")
            phrase = phrase.replace("scene of ", "")
            phrase = phrase.replace("- & footage", "")
            phrase = phrase.replace("& footage", "")
            phrase = phrase.strip()
            return phrase

        def group_similar_phrases_weighted(phrases, threshold=0.80):
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
            lowered = phrase.lower()
            if any(t in lowered for t in ["morning", "afternoon", "evening", "night", "sunset", "sunrise"]):
                return "time"
            if any(m in lowered for m in ["atmosphere", "mood", "feeling", "emotional", "ambience"]):
                return "mood"
            return "description"

        # CLIP similarity logging and semantic_caption generation (accept all valid captions)
        # Log CLIP similarity scores for each caption (but do not filter based on score)
        for src, text in raw_captions + all_captions:
            if not text or not selected_middle:
                continue
            try:
                image_input = preprocess_clip(selected_middle).unsqueeze(0).to(model_clip.device)
                text_input = clip.tokenize([text]).to(model_clip.device)
                with torch.no_grad():
                    image_features = model_clip.encode_image(image_input)
                    text_features = model_clip.encode_text(text_input)
                    similarity = (image_features @ text_features.T).item()
                logger.info(f"CLIP similarity for caption ({src}): {similarity:.2f} | {text}")
            except Exception as e:
                logger.warning(f"CLIP error on caption '{text}': {e}")

        # Accept all non-empty, valid, and non-duplicate captions
        accepted_phrases = [
            (src, normalize_phrase(text)) for src, text in raw_captions + all_captions
            if is_valid_caption(text) and not contains_unwanted_place(text) and len(text.split()) >= 4
        ]

        grouped_phrases = group_similar_phrases_weighted(accepted_phrases)
        descriptions = [p for p in grouped_phrases if classify_phrase(p) == "description"]
        times = [p for p in grouped_phrases if classify_phrase(p) == "time"]
        moods = [p for p in grouped_phrases if classify_phrase(p) == "mood"]
        if len(times) > 1:
            times = times[:1]
        final_order = descriptions + times + moods
        semantic_caption = " | ".join(final_order) if final_order else "No valid caption generated"

        frames_for_colors = [np.array(f) for f in selected_frames if f is not None]
        if frames_for_colors:
            stacked_frames = np.vstack([frame.reshape(-1, 3) for frame in frames_for_colors])
            kmeans = KMeans(n_clusters=3).fit(stacked_frames)
            colors = kmeans.cluster_centers_.astype(int)
            dominant_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in colors]
        else:
            dominant_colors = []

        motion_scores_list = [s["motion_score"] for s in segments]
        average_motion_score = float(round(np.mean(motion_scores_list), 3))
        motion_variance = float(round(np.var(motion_scores_list), 3))

        motion_tag = ""
        if average_motion_score >= 30:
            motion_tag = "dynamic"
        elif average_motion_score >= 15:
            motion_tag = "active"
        elif average_motion_score >= 5:
            motion_tag = "moderate"
        else:
            motion_tag = "still"


        commercial_junk = ["hd stock video", "royalty free", "stock footage", "high definition footage"]
        for junk in commercial_junk:
            semantic_caption = semantic_caption.replace(junk, "").strip()

        metadata_payload = {
            "semantic_caption": semantic_caption.lower(),
            "motion_score": average_motion_score,
            "motion_variance": motion_variance,
            "dominant_colors": dominant_colors,
            "motion_tag": motion_tag
        }

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
            "segments": segments,
            # "semantic_embedding": embedding,   # Removed semantic embedding generation
            "metadata_payload": metadata_payload
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"{video_id}.json"), "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Finished processing: {video_filename}")

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"Critical error while processing {video_filename}: {e}\n{tb_str}")
        with open(os.path.join(OUTPUT_DIR, "skipped_videos.log"), "a") as skipped_log:
            skipped_log.write(f"{video_path}, reason: CRITICAL ERROR: {e}\n")

# === Run Processing ===
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_all_videos(video_list):
    import signal

    executor = None  # placeholder to allow reference in shutdown handler

    def shutdown(signum, frame):
        print("Received shutdown signal, terminating workers...")
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    executor = ProcessPoolExecutor(max_workers=4)
    with executor:
        futures = []
        for filename in video_list:
            output_json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")
            # Only skip existing JSONs if not running in --only-flagged mode
            if not args.only_flagged and os.path.exists(output_json_path):
                logger.info(f"Skipping {filename}, already processed.")
                continue
            video_path = os.path.join(VIDEO_DIR, filename)
            futures.append(executor.submit(process_video, video_path))

        for idx, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                logger.info(f"[{idx}/{len(video_list)}] Finished processing one video.")
            except Exception as e:
                logger.error(f"Error processing a video: {e}")

if __name__ == "__main__":
    ensure_nltk_data()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Process only a random subset of videos for testing.")
    parser.add_argument("--from-file", type=str, help="Path to a JSON file containing list of video filenames to process.")
    parser.add_argument("--missing-json", action="store_true", help="Find videos in SCENES without corresponding JSON files in PROCESSED.")
    parser.add_argument("--sync", action="store_true", help="Process all videos in SCENES that do not have matching JSONs in PROCESSED.")
    parser.add_argument("--recover-from-originals", action="store_true", help="Attempt to reprocess original versions of quarantined ProRes files from another source directory.")
    parser.add_argument("--only-flagged", type=str, help="Path to a CSV file with video IDs to reprocess.")
    args, unknown = parser.parse_known_args()

    # --missing-json logic
    if args.missing_json:
        all_scene_videos = [
            f for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(VIDEO_EXTENSIONS) and not f.startswith("._")
        ]
        processed_jsons = {
            os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")
        }
        missing_json_videos = [
            f for f in all_scene_videos if os.path.splitext(f)[0] not in processed_jsons
        ]
        output_path = os.path.join(OUTPUT_DIR, "missing_json_videos.json")
        with open(output_path, "w") as f:
            json.dump(missing_json_videos, f, indent=2)
        print(f"Wrote {len(missing_json_videos)} entries to {output_path}")
        sys.exit(0)

    all_videos = [
        filename for filename in os.listdir(VIDEO_DIR)
        if filename.lower().endswith(VIDEO_EXTENSIONS) and not filename.startswith("._")
    ]

    # Improved logic to avoid misinterpreting argument flags as a video path
    if args.from_file:
        from_file_path = args.from_file
        if os.path.exists(from_file_path):
            logger.info(f"Using video list from file: {from_file_path}")
            with open(from_file_path, "r") as f:
                videos_to_process = json.load(f)
        else:
            logger.error(f"File not found: {from_file_path}")
            sys.exit(1)
    elif args.only_flagged:
        print("🧪 Running in ONLY-FLAGGED mode")
        flagged_path = args.only_flagged
        print("🧪 Path:", flagged_path)
        if os.path.exists(flagged_path):
            logger.info(f"Using flagged video list from CSV: {flagged_path}")
            with open(flagged_path, "r") as f:
                reader = csv.DictReader(f)
                videos_to_process = [row["video_id"] + ".mov" for row in reader if "video_id" in row]
        else:
            logger.error(f"File not found: {flagged_path}")
            sys.exit(1)
    elif args.test:
        test_file_path = os.path.join(os.path.dirname(OUTPUT_DIR), "test_videos.json")
        if os.path.exists(test_file_path):
            logger.info(f"Found existing test video list at {test_file_path}. Re-using it.")
            with open(test_file_path, "r") as f:
                test_videos = json.load(f)
        else:
            test_videos = random.sample(all_videos, min(100, len(all_videos)))
            logger.info(f"No existing test video list found. Selected {len(test_videos)} random videos.")
            with open(test_file_path, "w") as f:
                json.dump(test_videos, f, indent=2)
            logger.info(f"Saved selected test videos to {test_file_path}.")
        videos_to_process = test_videos
    elif args.sync:
        all_scene_videos = [
            f for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(VIDEO_EXTENSIONS) and not f.startswith("._")
        ]
        processed_jsons = {
            os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")
        }
        videos_to_process = [
            f for f in all_scene_videos if os.path.splitext(f)[0] not in processed_jsons
        ]
        logger.info(f"Sync mode: {len(videos_to_process)} videos need processing.")
    elif args.recover_from_originals:
        QUARANTINE_DIR = os.path.join(os.path.dirname(VIDEO_DIR), "QUARANTINE")
        ORIGINALS_DIR = "/Volumes/LaCie/RUNTIMEVIDEOS"
        all_quarantined = [
            f for f in os.listdir(QUARANTINE_DIR)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]
        base_ids = set()
        for qf in all_quarantined:
            base = qf.split("_")[0]
            if base:
                base_ids.add(base)
        original_candidates = []
        for f in os.listdir(ORIGINALS_DIR):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                for base in base_ids:
                    if f.startswith(base):
                        original_candidates.append(f)
                        break
        logger.info(f"Found {len(original_candidates)} original files to reprocess.")
        videos_to_process = original_candidates
        VIDEO_DIR = ORIGINALS_DIR
    elif len(sys.argv) > 1 and not args.test and not (sys.argv[1].startswith('--')):
        # If a positional argument is given and it's not a flag, treat as video path
        video_path = sys.argv[1]
        if os.path.isfile(video_path):
            process_video(video_path)
            sys.exit(0)
        else:
            logger.error(f"File not found: {video_path}")
            sys.exit(1)
    else:
        videos_to_process = all_videos

    print("🧪 Number of videos to process:", len(videos_to_process))
    print("🧪 First few videos:", videos_to_process[:5])
    process_all_videos(videos_to_process)