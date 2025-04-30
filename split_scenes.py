import time
import time as systime
def get_safe_thread_count():
    return 4
import os
import subprocess
import argparse
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect.video_stream import VideoOpenFailure

# Parameters
DEFAULT_THRESHOLD = 30.0
MIN_SCENE_LENGTH_SEC = 4.0  # minimum scene duration in seconds

def are_frames_similar(frame1, frame2, threshold=0.90):
    if frame1 is None or frame2 is None:
        return False
    frame1 = cv2.resize(frame1, (128, 128))
    frame2 = cv2.resize(frame2, (128, 128))
    frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([frame1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([frame2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > threshold

def grab_frame_at(video_path, timecode_sec, retries=3, delay=2.0):
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
    filtered = []
    last_frame = None

    for idx, (start, end) in enumerate(scene_list):
        duration = end.get_seconds() - start.get_seconds()
        if duration < MIN_SCENE_LENGTH_SEC:
            continue

        start_time_sec = start.get_seconds()
        frame = grab_frame_at(video_path, start_time_sec)

        if filtered:
            if last_frame is not None and are_frames_similar(last_frame, frame):
                continue  # Skip adding duplicate scene

        filtered.append((start, end))
        last_frame = frame

    return filtered

def detect_scenes(video_path, threshold=DEFAULT_THRESHOLD):
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

    # Fallback again: MotionDetector
    video_manager.release()
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    video_manager.release()
    return filter_short_scenes(scene_list, video_path)

from concurrent.futures import ThreadPoolExecutor

def split_video_by_scenes(video_path, scene_list, output_dir="scenes"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    def split_scene(i, start, end):
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

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(split_scene, i, start, end) for i, (start, end) in enumerate(scene_list)]
        for future in futures:
            future.result()

def main():
    parser = argparse.ArgumentParser(description="Split videos into scenes.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="ContentDetector threshold.")
    parser.add_argument("--input_dir", type=str, default="./converted", help="Directory with input videos.")
    parser.add_argument("--output_dir", type=str, default="scenes", help="Directory to save output scenes.")
    args = parser.parse_args()

    def is_already_split(video_file, output_dir):
        base_name = os.path.splitext(video_file)[0]
        expected_output = os.path.join(output_dir, f"{base_name}_scene_001.mov")
        return os.path.exists(expected_output)

    video_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".dv", ".avi", ".3gp", ".m4v", ".mxf"))
        and not f.startswith("._")
    ]

    video_files = [f for f in video_files if not is_already_split(f, args.output_dir)]

    total_files = len(video_files)
    print(f"\nFound {total_files} videos to process.\n")

    max_workers = get_safe_thread_count()

    def process_video(video_file):
        file_start_time = time.time()
        video_path = os.path.join(args.input_dir, video_file)
        print(f"\nProcessing: {video_file}")
        heartbeat_interval = 30  # seconds
        last_heartbeat = time.time()
        try:
            detect_start_time = time.time()
            scene_list = detect_scenes(video_path, threshold=args.threshold)
            detect_end_time = time.time()
            detect_elapsed = detect_end_time - detect_start_time
            print(f"  Scene detection for '{video_file}' took {detect_elapsed:.2f} seconds.")
            now = time.time()
            if now - last_heartbeat > heartbeat_interval:
                print(f"...still working on '{video_file}' (scene detection in progress)")
                last_heartbeat = now
        except VideoOpenFailure:
            print(f"  Skipping corrupt or unreadable file: {video_file}")
            return

        if not scene_list:
            output_dir = args.output_dir
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{base_name}_scene_001.mov")
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-c:v", "prores_ks", "-profile:v", "3",
                "-c:a", "aac",
                output_file
            ])
            print(f"  No scenes detected. Copied full video to: {output_file}")
            file_end_time = time.time()
            total_elapsed = file_end_time - file_start_time
            print(f"✅ Finished '{video_file}' in {total_elapsed:.2f} seconds.\n")
            return

        print(f"  Detected {len(scene_list)} scene(s) in '{video_file}' (after filtering):")
        for i, (start, end) in enumerate(scene_list):
            print(f"    {video_file} - Scene {i+1}: {start.get_timecode()} → {end.get_timecode()} ({end.get_seconds() - start.get_seconds():.2f} sec)")

        split_start_time = time.time()
        split_video_by_scenes(video_path, scene_list, output_dir=args.output_dir)
        split_end_time = time.time()
        split_elapsed = split_end_time - split_start_time
        print(f"  Scene splitting for '{video_file}' took {split_elapsed:.2f} seconds.")
        now = time.time()
        if now - last_heartbeat > heartbeat_interval:
            print(f"...still working on '{video_file}' (scene splitting in progress)")
            last_heartbeat = now

        file_end_time = time.time()
        total_elapsed = file_end_time - file_start_time
        print(f"✅ Finished '{video_file}' in {total_elapsed:.2f} seconds.\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video, f) for f in video_files]
        start_time = time.time()
        for idx, future in enumerate(as_completed(futures), 1):
            future.result()
            elapsed = time.time() - start_time
            avg_time_per_file = elapsed / idx
            remaining_files = total_files - idx
            eta_seconds = avg_time_per_file * remaining_files
            eta_minutes = int(eta_seconds // 60)
            eta_seconds_remainder = int(eta_seconds % 60)
            print(f"[{idx}/{total_files}] video(s) processed. ETA: {eta_minutes}m {eta_seconds_remainder}s remaining.")

if __name__ == "__main__":
    main()