import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime
import signal
import sys
import argparse

INPUT_DIR = "/Volumes/RUNTIME/JOSH CLIPS/FINAL FRANKIE CLIPS"
OUTPUT_DIR = "/Volumes/CORSAIR/THREAD2-HAP/FINAL CLIPS"
LOG_FILE = "conversion_errors.log"
MAX_WORKERS = 16
COOLDOWN_INTERVAL = 100
COOLDOWN_DURATION = 60

should_exit = False
futures = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_error(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | {msg}\n")

def handle_exit(signum, frame):
    global should_exit
    print("\nGracefully stopping... finishing current tasks.")
    should_exit = True

def convert_file(filename):
    if should_exit:
        return f"Skipped (interrupted): {filename}"
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = os.path.splitext(filename)[0] + "_hap.mov"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Resume check
    if os.path.exists(output_path):
        return f"Skipped (already exists): {filename}"

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", "scale=ceil(iw/4)*4:ceil(ih/4)*4,crop=trunc(iw/4)*4:trunc(ih/4)*4",
        "-c:v", "hap", "-format", "hap_q",
        "-video_track_timescale", "600",
        "-c:a", "aac",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        return f"Converted: {filename}"
    except subprocess.CalledProcessError as e:
        log_error(f"Failed: {filename} — {str(e)}")
        return f"Error: {filename}"

def main():
    global should_exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    parser = argparse.ArgumentParser(description="Convert ProRes to HAP")
    parser.add_argument("--dry-run", action="store_true", help="Only process the first 100 files for testing")
    args = parser.parse_args()

    all_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.mov', '.mp4', '.mxf')) and not f.startswith("._")
    ]
    if args.dry_run:
        all_files = all_files[:100]

    try:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            with tqdm(total=len(all_files)) as pbar:
                for i, filename in enumerate(all_files):
                    if should_exit:
                        break
                    future = executor.submit(convert_file, filename)
                    futures.append(future)

                    if (i + 1) % COOLDOWN_INTERVAL == 0:
                        time.sleep(COOLDOWN_DURATION)

                for future in as_completed(futures):
                    if should_exit:
                        break
                    print(future.result())
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\nCancelling remaining tasks...")
        for future in futures:
            future.cancel()

if __name__ == "__main__":
    main()