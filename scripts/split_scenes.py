"""
This script automatically splits video files into individual scenes based on visual content changes.
It uses multiple scene detection algorithms in sequence to ensure reliable scene detection:

1. ContentDetector: Primary detector that analyzes frame differences
2. ThresholdDetector: Fallback for videos with subtle content changes
3. AdaptiveDetector: Final fallback for challenging videos

The script includes features for:
- Parallel processing of multiple videos
- Minimum scene length enforcement
- Duplicate scene filtering
- Progress tracking and ETA calculation
- Automatic handling of corrupt files
- ProRes output format support
"""

import time
import time as systime
def get_safe_thread_count():
    """Return a safe number of threads for parallel processing."""
    return 4
import os
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect.video_stream import VideoOpenFailure
import config

# Configuration parameters
DEFAULT_THRESHOLD = 30.0  # Default threshold for content detection
MIN_SCENE_LENGTH_SEC = 4.0  # Minimum scene duration in seconds

def are_frames_similar(frame1, frame2, threshold=0.90):
    """
    Compare two frames for visual similarity using HSV color histograms.
    
    Args:
        frame1: First frame to compare
        frame2: Second frame to compare
        threshold: Similarity threshold (0-1)
        
    Returns:
        Boolean indicating if frames are similar
    """
    if frame1 is None or frame2 is None:
        return False
    
    # Resize frames for faster comparison
    frame1 = cv2.resize(frame1, (128, 128))
    frame2 = cv2.resize(frame2, (128, 128))
    
    # Convert to HSV color space
    frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    # Calculate and normalize histograms
    hist1 = cv2.calcHist([frame1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([frame2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    # Compare histograms
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > threshold

def grab_frame_at(video_path, timecode_sec, retries=3, delay=2.0):
    """
    Extract a single frame from a video at a specific timestamp.
    Includes retry logic for reliability.
    
    Args:
        video_path: Path to video file
        timecode_sec: Timestamp in seconds
        retries: Number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Frame image or None if extraction fails
    """
    for attempt in range(retries):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timecode_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        else:
            print(f"[WARN] Frame grab failed at {timecode_sec}s in '{os.path.basename(video_path)}'. Retrying ({attempt+1}/{retries})...")
            systime.sleep(delay)
    print(f"[ERROR] Failed to grab frame after {retries} attempts at {timecode_sec}s in '{os.path.basename(video_path)}'.")
    print(f"[TIMEOUT] File '{os.path.basename(video_path)}' triggered a read timeout at {timecode_sec:.2f}s.")
    return None

def filter_short_scenes(scene_list, video_path):
    """
    Filter out short scenes and visually similar consecutive scenes.
    
    Args:
        scene_list: List of (start, end) scene tuples
        video_path: Path to video file
        
    Returns:
        Filtered list of scenes
    """
    filtered = []
    last_frame = None

    for idx, (start, end) in enumerate(scene_list):
        # Skip scenes shorter than minimum duration
        duration = end.get_seconds() - start.get_seconds()
        if duration < MIN_SCENE_LENGTH_SEC:
            continue

        # Check for visual similarity with previous scene
        start_time_sec = start.get_seconds()
        frame = grab_frame_at(video_path, start_time_sec)

        if filtered:
            if last_frame is not None and are_frames_similar(last_frame, frame):
                continue  # Skip adding duplicate scene

        filtered.append((start, end))
        last_frame = frame

    return filtered

def detect_scenes(video_path, threshold=DEFAULT_THRESHOLD):
    """
    Detect scenes in a video using multiple detection algorithms.
    Falls back to simpler algorithms if primary detection fails.
    
    Args:
        video_path: Path to video file
        threshold: Detection threshold for ContentDetector
        
    Returns:
        List of detected scenes
    """
    video_manager = VideoManager([video_path])

    # First attempt: ContentDetector
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    if len(scene_list) > 1:
        video_manager.release()
        return filter_short_scenes(scene_list, video_path)

    # Fallback: ThresholdDetector
    video_manager.release()
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ThresholdDetector(threshold=12.0))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    if len(scene_list) > 1:
        video_manager.release()
        return filter_short_scenes(scene_list, video_path)

    # Final fallback: MotionDetector
    video_manager.release()
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    video_manager.release()
    return filter_short_scenes(scene_list, video_path)

def split_video_by_scenes(video_path, scene_list, output_dir=None):
    """
    Split a video into individual scene files.
    
    Args:
        video_path: Path to input video
        scene_list: List of detected scenes
        output_dir: Directory for output files (defaults to config.SCENES_DIR)
    """
    if output_dir is None:
        output_dir = config.SCENES_DIR
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    def split_scene(i, start, end):
        """Split a single scene from the video."""
        start_time = start.get_timecode()
        duration = end.get_seconds() - start.get_seconds()
        output_file = os.path.join(output_dir, f"{base_name}_scene_{i+1:03d}.mov")

        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-ss", start_time, "-t", str(duration),
            "-c", "copy",
            output_file
        ])
        print(f"Saved: {output_file}")

    # Process scenes in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(split_scene, i, start, end) for i, (start, end) in enumerate(scene_list)]
        for future in futures:
            future.result()

def main():
    """
    Main entry point for video scene splitting.
    Processes all ProRes videos in the configured directory.
    """
    def is_already_split(video_file, output_dir):
        """Check if a video has already been processed."""
        base_name = os.path.splitext(video_file)[0]
        expected_output = os.path.join(output_dir, f"{base_name}_scene_001.mov")
        return os.path.exists(expected_output)

    # Find videos to process
    video_files = [
        f for f in os.listdir(config.PRORES_DIR)
        if f.lower().endswith(tuple(ext.lower() for ext in config.VIDEO_EXTENSIONS))
        and not f.startswith("._")
        and "_prores" in f.lower()  # Only process ProRes files
    ]

    # Filter out already processed videos
    video_files = [f for f in video_files if not is_already_split(f, config.SCENES_DIR)]

    total_files = len(video_files)
    print(f"\nFound {total_files} videos to process.\n")

    max_workers = get_safe_thread_count()

    def process_video(video_file):
        """Process a single video file."""
        file_start_time = time.time()
        video_path = os.path.join(config.PRORES_DIR, video_file)
        print(f"\nProcessing: {video_file}")
        heartbeat_interval = 30  # seconds

        try:
            # Detect scenes
            scene_list = detect_scenes(video_path)
            
            if not scene_list:
                print(f"No scenes detected in {video_file}, creating single scene file...")
                # Create output directory if it doesn't exist
                os.makedirs(config.SCENES_DIR, exist_ok=True)
                
                # Get video duration using ffprobe
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", video_path
                ]
                duration = float(subprocess.check_output(duration_cmd).decode().strip())
                
                # Create a single scene file
                output_file = os.path.join(config.SCENES_DIR, f"{os.path.splitext(video_file)[0]}_scene_001.mov")
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path,
                    "-c", "copy",
                    output_file
                ])
                print(f"Created single scene file: {output_file}")
            else:
                # Split video into scenes
                split_video_by_scenes(video_path, scene_list)
            
            # Print processing time
            processing_time = time.time() - file_start_time
            print(f"Completed {video_file} in {processing_time:.1f} seconds")
            
        except VideoOpenFailure as e:
            print(f"Error opening {video_file}: {str(e)}")
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")

    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video, video_file) for video_file in video_files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker thread: {str(e)}")

    print("\nScene splitting complete!")
    print(f"Processed {total_files} videos")
    print(f"Output scenes saved to: {config.SCENES_DIR}")
    print("\nNext Steps:")
    print("1. Review the generated scenes in the scenes directory")
    print("2. Run the next step in the pipeline:")
    print("   python scripts/process_videos.py")

if __name__ == "__main__":
    main()