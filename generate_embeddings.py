# generate_embeddings.py
import os
import json
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        joined_text = " ".join([
            data.get("ai_description", ""),
            " ".join(data.get("semantic_tags", [])),
            " ".join(data.get("formal_tags", [])),
            " ".join(data.get("emotional_tags", [])),
            " ".join(data.get("material_tags", [])),
        ])

        embedding = embed_model.encode(joined_text, convert_to_tensor=True).cpu().numpy().tolist()
        data["semantic_embedding"] = embedding

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Generated embedding for {filename}")
    except Exception as e:
        print(f"Failed embedding {filename}: {e}")