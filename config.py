"""
Configuration settings for the runtime performance system.
Centralizes path configurations and other settings for easy modification.

Directory Structure:
------------------
./                      # Project root directory
├── data/              # All data directories
│   ├── source/       # Original source video files
│   ├── prores/       # ProRes converted videos
│   ├── playback/     # HAP encoded videos
│   ├── scenes/       # Scene-split videos and metadata
│   │   └── thumbnails/  # Video thumbnails
│   └── descriptions/ # AI-generated descriptions and embeddings
└── runtime.toe       # TouchDesigner project file

Required Permissions:
-------------------
- All directories must be writable
- TouchDesigner project file must be readable
- Source video files must be readable
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent  # Set to the actual project root directory
DATA_DIR = PROJECT_ROOT / "data"

# Video processing directories
SOURCE_DIR = DATA_DIR / "source"          # Original source videos
PRORES_DIR = DATA_DIR / "prores"          # ProRes converted videos
PLAYBACK_DIR = DATA_DIR / "playback"      # HAP encoded videos
SCENES_DIR = DATA_DIR / "scenes"          # Scene-split videos and metadata
THUMBNAILS_DIR = SCENES_DIR / "thumbnails"  # Video thumbnails

# Output directories
OUTPUT_DIR = DATA_DIR / "descriptions"  # AI-generated descriptions

# Video processing settings
PRORES_PROFILE = "ProRes422"  # Options: ProRes422, ProRes422HQ, ProRes4444
HAP_QUALITY = "high"  # Options: low, medium, high
VIDEO_EXTENSIONS = [".mov", ".mp4", ".avi", ".mkv", ".dv"]
THUMBNAIL_SIZE = (320, 180)  # Width, Height in pixels
SCENE_THRESHOLD = 30  # Minimum frames between scene cuts

# AI settings
BLIP_MODEL = "Salesforce/blip2-opt-2.7b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CAPTION_LENGTH = 100
MIN_CAPTION_LENGTH = 5
GPT_MODEL = "gpt-4"  # Options: gpt-4, gpt-3.5-turbo
TEMPERATURE = 0.7  # AI generation temperature (0.0 to 1.0)

# Caption regeneration settings
MAX_REGENERATION_ATTEMPTS = 1  # Maximum number of attempts to regenerate a caption before removing it

# Performance settings
MOVEMENTS = [
    "orientation",
    "elemental",
    "built",
    "people",
    "blur"
]

# Logging settings
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = PROJECT_ROOT / "runtime.log"

# Ensure directories exist
for directory in [SOURCE_DIR, PRORES_DIR, PLAYBACK_DIR, SCENES_DIR, 
                 THUMBNAILS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Validation
def validate_config():
    """Validate that all required paths and settings are properly configured."""
    if not PROJECT_ROOT.exists():
        raise ValueError(f"Project root directory does not exist: {PROJECT_ROOT}")
    
    if not all(isinstance(movement, str) for movement in MOVEMENTS):
        raise ValueError("All movements must be strings")
    
    if PRORES_PROFILE not in ["ProRes422", "ProRes422HQ", "ProRes4444"]:
        raise ValueError(f"Invalid ProRes profile: {PRORES_PROFILE}")
    
    if HAP_QUALITY not in ["low", "medium", "high"]:
        raise ValueError(f"Invalid HAP quality: {HAP_QUALITY}")
    
    if not isinstance(THUMBNAIL_SIZE, tuple) or len(THUMBNAIL_SIZE) != 2:
        raise ValueError("THUMBNAIL_SIZE must be a tuple of (width, height)")
    
    if not 0 <= TEMPERATURE <= 1:
        raise ValueError("TEMPERATURE must be between 0 and 1")
    
    if LOG_LEVEL not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid LOG_LEVEL: {LOG_LEVEL}")

# Run validation on import
validate_config() 