import os
import subprocess
import argparse
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Parameters
DEFAULT_THRESHOLD = 30.0
MIN_SCENE_LENGTH_SEC = 2.0  # minimum scene duration in seconds

def detect_scenes(video_path, threshold=DEFAULT_THRESHOLD):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    # Filter out very short scenes
    filtered_scenes = [
        (start, end) for start, end in scene_list
        if (end.get_seconds() - start.get_seconds()) >= MIN_SCENE_LENGTH_SEC
    ]

    return filtered_scenes

def split_video_by_scenes(video_path, scene_list, output_dir="scenes"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i, (start, end) in enumerate(scene_list):
        start_time = start.get_timecode()
        duration = end.get_seconds() - start.get_seconds()
        output_file = os.path.join(output_dir, f"{base_name}_scene_{i+1:03d}.mov")

        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-ss", start_time, "-t", str(duration),
            "-vf", "scale=trunc(iw/4)*4:trunc(ih/4)*4",
            "-c:v", "prores_ks", "-profile:v", "3",
            "-c:a", "aac",
            output_file
        ])
        print(f"Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Split videos into scenes.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="ContentDetector threshold.")
    parser.add_argument("--input_dir", type=str, default="./converted", help="Directory with input videos.")
    args = parser.parse_args()

    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith((".mp4", ".mov", ".mkv", ".dv", ".avi", ".3gp", ".m4v", ".mxf"))]

    for video_file in video_files:
        video_path = os.path.join(args.input_dir, video_file)
        print(f"\nProcessing: {video_file}")
        scene_list = detect_scenes(video_path, threshold=args.threshold)

        if not scene_list:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = "scenes"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{base_name}_scene_001.mov")
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-c:v", "prores_ks", "-profile:v", "3",
                "-c:a", "aac",
                output_file
            ])
            print(f"  No scenes detected. Copied full video to: {output_file}")
            continue

        print(f"  Detected {len(scene_list)} scenes (after filtering):")
        for i, (start, end) in enumerate(scene_list):
            print(f"    Scene {i+1}: {start.get_timecode()} → {end.get_timecode()} ({end.get_seconds() - start.get_seconds():.2f} sec)")

        split_video_by_scenes(video_path, scene_list)

if __name__ == "__main__":
    main()