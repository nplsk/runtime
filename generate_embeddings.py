"""
This script generates semantic embeddings for video metadata using the SentenceTransformer model.
It processes JSON files containing video descriptions and tags, combining them into a single
text representation that is then converted into a high-dimensional vector embedding.

The embeddings capture the semantic meaning of:
- AI-generated descriptions
- Semantic tags
- Formal tags (composition, lighting, etc.)
- Emotional tags
- Material tags

These embeddings can be used for:
- Semantic similarity search
- Clustering similar videos
- Finding related content
- Content recommendation

Input:
- Directory of JSON files containing video metadata
- Each file should contain description and tag fields

Output:
- Updated JSON files with added "semantic_embedding" field
- Each embedding is a 384-dimensional vector (using all-MiniLM-L6-v2 model)
"""

import os
import json
from sentence_transformers import SentenceTransformer

# Directory containing processed video metadata files
OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"

# Initialize the sentence transformer model
# all-MiniLM-L6-v2 is a good balance of speed and quality
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Process each JSON file in the directory
for filename in os.listdir(OUTPUT_DIR):
    # Skip non-JSON files and macOS metadata files
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
        
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        # Load the video metadata
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Combine all text fields into a single string
        # This creates a comprehensive representation of the video's content
        joined_text = " ".join([
            data.get("ai_description", ""),
            " ".join(data.get("semantic_tags", [])),
            " ".join(data.get("formal_tags", [])),
            " ".join(data.get("emotional_tags", [])),
            " ".join(data.get("material_tags", [])),
        ])

        # Generate embedding and convert to list format for JSON storage
        embedding = embed_model.encode(joined_text, convert_to_tensor=True).cpu().numpy().tolist()
        data["semantic_embedding"] = embedding

        # Save the updated metadata with the new embedding
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Generated embedding for {filename}")
    except Exception as e:
        print(f"Failed embedding {filename}: {e}")