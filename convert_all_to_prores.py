import os
import subprocess

INPUT_DIR = "./videos"
OUTPUT_DIR = "./prores_converted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_convertible(filename):
    # Skip already converted ProRes or HAP files
    return (filename.endswith((".mp4", ".mov")) and
            "prores" not in filename.lower() and
            "hap" not in filename.lower())

def convert_to_prores(input_file, output_file):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_file,
        "-c:v", "prores_ks", "-profile:v", "3",
        "-af", "loudnorm",
        "-c:a", "aac",
        output_file
    ]
    subprocess.run(command)

for root, _, files in os.walk(INPUT_DIR):
    for filename in files:
        if is_convertible(filename):
            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(input_path, INPUT_DIR)
            output_filename = os.path.splitext(rel_path)[0] + "_prores.mov"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Converting {input_path} → {output_path}")
            convert_to_prores(input_path, output_path)
