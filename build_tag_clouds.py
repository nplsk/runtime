# build_tag_clouds.py
import os
import json
from collections import defaultdict, Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"
OUTPUT_DIR = "./tag_clouds"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cluster_tags = defaultdict(list)

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
        for key in ["semantic_tags", "formal_tags", "emotional_tags", "material_tags"]:
            tags.extend(data.get(key, []))

        mood = data.get("mood_tag", "")
        if mood:
            tags.append(mood)
        motion = data.get("motion_tag", "")
        if motion:
            tags.append(motion)

        cluster_tags[cluster_label].extend(tags)

    except Exception as e:
        print(f"Failed reading {filename}: {e}")

# Generate word clouds
for cluster_label, tags in cluster_tags.items():
    freq = Counter(tags)
    wc = WordCloud(width=800, height=600, background_color="white", colormap="plasma").generate_from_frequencies(freq)

    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Cluster {cluster_label} Tag Cloud")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_{cluster_label}_tagcloud.png"))
    plt.close()

print("✅ Tag clouds saved per cluster in tag_clouds/ directory.")