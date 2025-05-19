"""
Find similar videos based on semantic similarity using embeddings.
"""

import os
import sys
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import config
from config import (
    OUTPUT_DIR,
    EMBEDDING_MODEL
)

# Minimum similarity threshold (0.0 to 1.0)
MIN_SIMILARITY_THRESHOLD = 0.3

def load_video_embeddings():
    """Load all video embeddings from JSON files."""
    videos = []
    for json_path in OUTPUT_DIR.glob("*.json"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'metadata_payload' in data and 'embedding' in data['metadata_payload']:
                    videos.append({
                        'id': data['video_id'],
                        'description': data['metadata_payload'].get('ai_description', ''),
                        'embedding': torch.tensor(data['metadata_payload']['embedding'], dtype=torch.float32),
                        'tags': {
                            'semantic': data['metadata_payload'].get('semantic_tags', []),
                            'formal': data['metadata_payload'].get('formal_tags', []),
                            'emotional': data['metadata_payload'].get('emotional_tags', []),
                            'material': data['metadata_payload'].get('material_tags', [])
                        }
                    })
        except Exception as e:
            print(f"Error loading {json_path}: {str(e)}")
    return videos

def find_similar_videos(query, videos, top_k=5):
    """Find videos similar to the query text."""
    # Load the model
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate embedding for the query
    query_embedding = torch.tensor(model.encode(query), dtype=torch.float32)
    
    # Calculate similarities
    similarities = []
    for video in videos:
        similarity = util.pytorch_cos_sim(
            query_embedding.unsqueeze(0),
            video['embedding'].unsqueeze(0)
        ).item()
        if similarity >= MIN_SIMILARITY_THRESHOLD:
            similarities.append((video, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def main():
    # Load all video embeddings
    print("Loading video embeddings...")
    videos = load_video_embeddings()
    print(f"Loaded {len(videos)} videos")
    print(f"Minimum similarity threshold: {MIN_SIMILARITY_THRESHOLD}")
    
    while True:
        # Get query from user
        query = input("\nEnter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        # Find similar videos
        print(f"\nSearching for videos similar to: {query}")
        similar_videos = find_similar_videos(query, videos)
        
        # Display results
        if similar_videos:
            print("\nMost similar videos:")
            for i, (video, similarity) in enumerate(similar_videos, 1):
                print(f"\n{i}. {video['id']} (Similarity: {similarity:.3f})")
                print(f"   Description: {video['description']}")
                print("   Tags:")
                for category, tags in video['tags'].items():
                    if tags:
                        print(f"   - {category}: {', '.join(tags)}")
        else:
            print(f"\nNo videos found with similarity above {MIN_SIMILARITY_THRESHOLD}")

if __name__ == "__main__":
    main() 