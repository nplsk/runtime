# build_cluster_manifest.py
import os
import json
from collections import defaultdict, Counter

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"
OUTPUT_FILE = "cluster_manifest.txt"

cluster_data = defaultdict(list)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cluster_label = data.get("cluster_label", None)
        if cluster_label is None:
            continue

        tags = []
        for tag_key in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
            tags.extend(data.get(tag_key, []))

        if mood := data.get("mood_tag"):
            tags.append(mood)
        if motion := data.get("motion_tag"):
            tags.append(motion)

        cluster_data[cluster_label].extend(tags)

    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Write manifest
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for cluster_id in sorted(cluster_data):
        tags = cluster_data[cluster_id]
        freq = Counter(tags).most_common(10)
        tag_summary = ", ".join([f"{tag} ({count})" for tag, count in freq])
        out.write(f"Cluster {cluster_id}:\n")
        out.write(f"  Top Tags: {tag_summary}\n\n")

print(f"✅ Wrote cluster summary to {OUTPUT_FILE}")