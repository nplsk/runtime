import os
import json
import numpy as np
import plotly.graph_objects as go
import umap
import pandas as pd
import base64

INPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"
THUMBNAIL_DIR = "/Volumes/RUNTIME/PROCESSED/thumbnails"
OUTPUT_HTML = "semantic_map_thumbnails.html"

filenames = []
embeddings = []
cluster_labels = []
mood_tags = []
motion_tags = []
sample_tags = []
motion_scores = []
encoded_thumbnails = []

# Load JSON files
def encode_thumbnail(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

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
            cluster_labels.append(str(data.get("cluster_label", "null")))
            mood_tags.append(data.get("mood_tag", ""))
            motion_tags.append(data.get("motion_tag", ""))
            sample_tags.append(", ".join(data.get("semantic_tags", [])[:5]))
            motion_scores.append(float(data.get("motion_score", 0)))

            base_name = os.path.splitext(filename)[0]
            thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{base_name}.jpg")
            encoded_thumbnails.append(encode_thumbnail(thumbnail_path))

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
    "tags": sample_tags,
    "motion_score": motion_scores
})

# Normalize motion_score for size scaling
min_size = 10
max_size = 40
if df["motion_score"].max() > 0:
    df["size"] = min_size + (df["motion_score"] / df["motion_score"].max()) * (max_size - min_size)
else:
    df["size"] = min_size

# Create base scatter plot
fig = go.Figure()

for i, row in df.iterrows():
    thumb_encoded = encoded_thumbnails[i]
    if thumb_encoded:
        fig.add_layout_image(
            dict(
                source=thumb_encoded,
                x=row["x"],
                y=row["y"],
                sizex=0.5,  # Control thumbnail size here
                sizey=0.5,
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                sizing="stretch",
                opacity=0.95,
                layer="below"
            )
        )

# Add invisible scatter points for hover interaction
# Add barely-visible scatter points for hover interaction
fig.add_trace(go.Scatter(
    x=df["x"],
    y=df["y"],
    mode="markers",
    marker=dict(
        size=df["size"] + 5,            # Boost size a little for better hover targeting
        color="rgba(0,0,0,0.001)",       # Almost invisible but "real" hover area
        line=dict(width=0)
    ),
    hovertemplate=(
        "<b>%{customdata[0]}</b><br><br>" +
        "Cluster: %{customdata[1]}<br>" +
        "Mood: %{customdata[2]}<br>" +
        "Motion: %{customdata[3]}<br>" +
        "Tags: %{customdata[4]}<br><extra></extra>"
    ),
    customdata=np.stack([df["filename"], df["cluster"], df["mood"], df["motion"], df["tags"]], axis=-1)
))

# Layout
fig.update_layout(
    title="Video Archive Semantic Map with Thumbnails",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    width=1400,
    height=1000,
    margin=dict(l=0, r=0, t=50, b=0),
    hovermode="closest",
)

fig.write_html(OUTPUT_HTML)
print(f"✅ Saved interactive thumbnail map to {OUTPUT_HTML}")