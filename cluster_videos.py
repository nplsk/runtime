# cluster_videos.py
import os
import json
import numpy as np
import hdbscan
from collections import defaultdict

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"

# Load all JSONs with embeddings
filepaths = []
embeddings = []

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        emb = data.get("semantic_embedding", [])
        if emb and isinstance(emb, list) and len(emb) > 0:
            filepaths.append(path)
            embeddings.append(emb)

print(f"Loaded {len(embeddings)} files with embeddings.")

# Run HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean')
labels = clusterer.fit_predict(np.array(embeddings))

# Apply labels and write back to files
cluster_counts = defaultdict(int)
for path, label in zip(filepaths, labels):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["cluster_label"] = int(label) if label >= 0 else None
    if label >= 0:
        cluster_counts[int(label)] += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Report summary
print("\nCluster summary:")
for cluster_id, count in sorted(cluster_counts.items()):
    print(f"  Cluster {cluster_id}: {count} videos")

print("✅ Assigned cluster labels using HDBSCAN.")