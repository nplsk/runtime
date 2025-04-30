# visualize_proximity.py
import os
import json
import numpy as np
import plotly.express as px
import umap
import pandas as pd

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"

filenames = []
embeddings = []
cluster_labels = []
mood_tags = []
motion_tags = []
sample_tags = []

# Load JSON files
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        embedding = data.get("semantic_embedding", [])
        if embedding and isinstance(embedding, list):
            embeddings.append(embedding)
            filenames.append(filename)
            cluster_labels.append(data.get("cluster_label", "null"))
            mood_tags.append(data.get("mood_tag", ""))
            motion_tags.append(data.get("motion_tag", ""))
            sample_tags.append(", ".join(data.get("semantic_tags", [])[:5]))
    except Exception as e:
        print(f"Failed reading {filename}: {e}")

# Reduce embeddings to 2D
print(f"Reducing {len(embeddings)} embeddings to 2D...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(np.array(embeddings))

# Prepare DataFrame
df = pd.DataFrame({
    "x": embedding_2d[:, 0],
    "y": embedding_2d[:, 1],
    "filename": filenames,
    "cluster": cluster_labels,
    "mood": mood_tags,
    "motion": motion_tags,
    "tags": sample_tags
})

# Plot
fig = px.scatter(
    df,
    x="x", y="y",
    color="cluster",
    hover_data=["filename", "cluster", "mood", "motion", "tags"],
    title="Video Archive Semantic Map (by Cluster)",
    width=1000, height=800
)

fig.update_layout(legend_title_text='Cluster')
fig.write_html("semantic_map.html")

print("✅ Saved interactive map to semantic_map.html")