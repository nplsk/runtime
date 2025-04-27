import os
import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd

CLUSTER_LABELS = {
    0: "elemental stillness",
    1: "first perception",
    2: "human proximity",
    3: "residual interference",
    4: "luminous threshold"
}

INPUT_DIR = "./output"

# Load data
vectors = []
labels = []
filenames = []

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".json"):
        with open(os.path.join(INPUT_DIR, fname)) as f:
            data = json.load(f)
            embedding = data.get("semantic_embedding", [])
            if not embedding or "cluster" not in data:
                continue
            # Append motion and brightness if present
            segments = data.get("segments", [])
            avg_motion = np.mean([s["motion_score"] for s in segments]) / 100 if segments else 0.0
            brightness = 0.5
            try:
                from PIL import Image
                img = Image.open(segments[0]["thumbnail"]).convert("RGB")
                colors = img.getcolors(maxcolors=256000)
                brightness = sum([sum(c[1]) for c in colors]) / (len(colors) * 3) / 255.0
            except:
                pass
            vectors.append(embedding + [avg_motion, brightness])
            labels.append(data["cluster"])
            filenames.append(fname)

# UMAP
X = np.array(vectors)
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)

import plotly.express as px

df = pd.DataFrame(embedding, columns=["x", "y"])
df["cluster"] = labels
df["filename"] = filenames
df["cluster_name"] = df["cluster"].map(CLUSTER_LABELS)

# Compute centroids
centroids = df.groupby("cluster_name")[["x", "y"]].mean().reset_index()
centroids["filename"] = "CENTROID"
centroids["hover"] = centroids["cluster_name"]

# Add hover info
df["hover"] = df["filename"] + " — " + df["cluster_name"]

# Combine points and centroids
combined_df = pd.concat([df, centroids], ignore_index=True)

fig = px.scatter(
    combined_df,
    x="x", y="y",
    color="cluster_name",
    hover_name="hover",
    title="Clustered Archive UMAP (Interactive)",
    labels={
        "x": "UMAP Dimension 1 (semantic + visual similarity)",
        "y": "UMAP Dimension 2 (semantic + visual similarity)",
        "cluster_name": "Cluster"
    },
    width=1000, height=800
)

fig.update_traces(marker=dict(size=8))
fig.show()

df.to_csv("clustered_umap_output.csv", index=False)