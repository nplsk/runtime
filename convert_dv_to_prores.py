import os
import subprocess

INPUT_DIR = "./videos"
OUTPUT_DIR = "./converted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_files = []
for f in os.listdir(INPUT_DIR):
    if os.path.splitext(f)[1].lower() in [".dv", ".avi"]:
        video_files.append(f)

for filename in video_files:
    input_path = os.path.join(INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.mov")

    print(f"Converting {filename} → {output_path}")

    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", "scale=853:480,setdar=16/9",
        "-c:v", "prores_ks", "-profile:v", "3",  # ProRes 422 HQ
        "-c:a", "pcm_s16le",  # Uncompressed audio
        output_path
    ])

    print(f"Saved: {output_path}")