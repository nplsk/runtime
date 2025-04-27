import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from PIL import Image
import cv2

# CONFIG
INPUT_DIR = "./output"
NUM_CLUSTERS = 5
CLUSTER_LABELS = {
    0: "elemental stillness",
    1: "first perception",
    2: "human proximity",
    3: "residual interference",
    4: "luminous threshold"
}

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def estimate_brightness(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        colors = img.getcolors(maxcolors=256000)
        brightness = sum([sum(c[1]) for c in colors]) / (len(colors) * 3)
        return brightness / 255.0  # normalize to 0-1
    except:
        return 0.5  # fallback

# Load descriptions
paths = []
descriptions = []

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".json"):
        path = os.path.join(INPUT_DIR, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
                tag_fields = ["semantic_tags", "formal_tags", "emotional_tags", "time_tags", "material_tags"]
                all_tags = sum([data.get(field, []) for field in tag_fields], [])
                desc = " ".join(all_tags) or data.get("ai_description", "") or data.get("description", "")
                if not desc:
                    print(f"No tags or description found in {fname}")
                    continue

                # Compute average motion score
                segments = data.get("segments", [])
                avg_motion = np.mean([s["motion_score"] for s in segments]) / 100 if segments else 0.0

                # Estimate brightness from first thumbnail
                thumbnail_path = segments[0]["thumbnail"] if segments else ""
                brightness = estimate_brightness(thumbnail_path)

                # Get sentence embedding
                embedding = model.encode(desc)

                # Build extended vector
                extended = np.concatenate([embedding, [avg_motion, brightness]])
                paths.append(path)
                descriptions.append(extended)
        except Exception as e:
            print(f"Error loading {path}: {e}")

if not descriptions:
    print("No descriptions found. Exiting.")
    exit()

# Generate embeddings
print("Encoding tags...")
X = np.array(descriptions)

# Run clustering
print(f"Clustering into {NUM_CLUSTERS} groups...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

# Write back to JSON files
print("Writing cluster labels...")
for path, label in tqdm(zip(paths, labels), total=len(paths)):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        data["cluster"] = int(label)
        data["cluster_name"] = CLUSTER_LABELS[label]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating {path}: {e}")

print("✅ Clustering complete.")