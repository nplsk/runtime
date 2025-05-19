"""
Generates semantic embeddings for video descriptions.
Uses paths and settings from config.py.
"""

import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

print("Starting script...")

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

import config
from config import (
    OUTPUT_DIR,
    EMBEDDING_MODEL
)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Embedding model: {EMBEDDING_MODEL}")

def generate_embedding(metadata_path):
    """Generate embedding for a single video's metadata."""
    try:
        print(f"\nProcessing file: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("Successfully loaded metadata")

        # Use metadata_payload for embedding
        payload = metadata.get('metadata_payload', {})
        text_parts = []
        # Use ai_description as the main summary
        if 'ai_description' in payload:
            text_parts.append(payload['ai_description'])
        # Add all tag categories
        for tag_cat in ['semantic_tags', 'formal_tags', 'emotional_tags', 'material_tags']:
            if tag_cat in payload and isinstance(payload[tag_cat], list):
                text_parts.extend(payload[tag_cat])

        if not text_parts:
            print(f"No text content found in {metadata_path}")
            return None

        # Combine text parts
        text = " ".join(text_parts)
        print(f"Text content: {text}")

        # Generate embedding
        print("Loading sentence transformer model...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("Generating embedding...")
        embedding = model.encode(text)
        print("Embedding generated successfully")

        # Update metadata with embedding in metadata_payload
        payload['embedding'] = embedding.tolist()
        metadata['metadata_payload'] = payload

        # Save updated metadata
        print("Saving updated metadata...")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Successfully saved embedding for {metadata_path}")

        return metadata

    except Exception as e:
        print(f"Error generating embedding for {metadata_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Generate embeddings for all video metadata files."""
    print("\nStarting main function...")
    
    # Get list of metadata files
    metadata_files = list(OUTPUT_DIR.glob("*.json"))
    print(f"Found {len(metadata_files)} JSON files")
    
    if not metadata_files:
        print(f"No metadata files found in {OUTPUT_DIR}")
        return

    # Generate embeddings for each file
    print(f"Generating embeddings for {len(metadata_files)} files...")
    for metadata_path in tqdm(metadata_files, desc="Generating embeddings"):
        generate_embedding(metadata_path)

if __name__ == "__main__":
    main()