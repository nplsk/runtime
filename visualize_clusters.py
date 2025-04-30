# visualize_clusters.py
import os
import json
import random
import csv
from collections import defaultdict

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"
SAMPLES_PER_CLUSTER = 5
OUTPUT_CSV = "cluster_samples.csv"

cluster_groups = defaultdict(list)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cluster_label = data.get("cluster_label", None)
        if cluster_label is not None:
            video_filename = os.path.basename(data.get("file_path", ""))
            cluster_groups[cluster_label].append({
                "filename": filename,
                "video_filename": video_filename,
                "ai_description": data.get("ai_description", ""),
                "semantic_tags": data.get("semantic_tags", []),
                "mood_tag": data.get("mood_tag", ""),
                "motion_tag": data.get("motion_tag", ""),
            })

    except Exception as e:
        print(f"Failed reading {filename}: {e}")

with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["cluster_label", "json_filename", "video_filename", "mood_tag", "motion_tag", "sample_tags", "short_description"])

    for cluster_label, entries in sorted(cluster_groups.items()):
        samples = random.sample(entries, min(SAMPLES_PER_CLUSTER, len(entries)))
        for entry in samples:
            sample_tags = ", ".join(entry["semantic_tags"][:5])
            short_description = entry["ai_description"][:100] + "..." if len(entry["ai_description"]) > 100 else entry["ai_description"]
            writer.writerow([
                cluster_label,
                entry["filename"],
                entry["video_filename"],
                entry["mood_tag"],
                entry["motion_tag"],
                sample_tags,
                short_description
            ])

print(f"✅ Updated cluster samples saved to {OUTPUT_CSV}.")