"""
Generates semantic embeddings for video descriptions.
Uses paths and settings from config.py.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import config

def generate_embedding(metadata_path):
    """Generate embedding for a single video's metadata."""
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Get text to embed
        text_parts = []
        if 'description' in metadata:
            text_parts.append(metadata['description'])
        if 'tags' in metadata:
            text_parts.extend(metadata['tags'])
            
        if not text_parts:
            print(f"No text content found in {metadata_path}")
            return None
            
        # Combine text parts
        text = " ".join(text_parts)
        
        # Generate embedding
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        embedding = model.encode(text)
        
        # Update metadata with embedding
        metadata['embedding'] = embedding.tolist()
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
        
    except Exception as e:
        print(f"Error generating embedding for {metadata_path}: {str(e)}")
        return None

def main():
    """Generate embeddings for all video metadata files."""
    # Get list of metadata files
    metadata_files = list(config.SCENES_DIR.glob("*.json"))
    
    if not metadata_files:
        print(f"No metadata files found in {config.SCENES_DIR}")
        return

    # Generate embeddings for each file
    print(f"Generating embeddings for {len(metadata_files)} files...")
    for metadata_path in tqdm(metadata_files, desc="Generating embeddings"):
        generate_embedding(metadata_path)

if __name__ == "__main__":
    main()