import os
import subprocess

# INPUT_DIR = "./scenes"
# OUTPUT_DIR = "./scenes_hap"

INPUT_DIR = os.path.expanduser("~/Downloads/grid-test")
OUTPUT_DIR = os.path.expanduser("~/Downloads/grid-test/hap")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_to_hap(input_file, output_file):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_file,
        "-vf", "scale=ceil(iw/4)*4:ceil(ih/4)*4,crop=trunc(iw/4)*4:trunc(ih/4)*4",
        "-c:v", "hap", "-format", "hap_q",
        "-video_track_timescale", "600",
        "-c:a", "aac",
        output_file
    ]
    subprocess.run(command)

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".MOV") or filename.endswith(".mp4"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + "_hap.mov"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        print(f"Converting {input_path} → {output_path}")
        convert_to_hap(input_path, output_path)
