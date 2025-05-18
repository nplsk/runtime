"""
Configuration settings for the runtime performance system.
Centralizes path configurations and other settings for easy modification.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path("/Volumes/RUNTIME")
PROCESSED_DIR = BASE_DIR / "PROCESSED"
PROCESSED_DESCRIPTIONS_DIR = BASE_DIR / "PROCESSED_DESCRIPTIONS"

# Input/Output paths
INPUT_DIR = PROCESSED_DIR
OUTPUT_DIR = PROCESSED_DESCRIPTIONS_DIR

# Video processing settings
PRORES_PROFILE = "ProRes422"  # Options: ProRes422, ProRes422HQ, ProRes4444
HAP_QUALITY = "high"  # Options: low, medium, high

# AI settings
BLIP_MODEL = "Salesforce/blip2-opt-2.7b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CAPTION_LENGTH = 100
MIN_CAPTION_LENGTH = 5

# Performance settings
MOVEMENTS = [
    "orientation",
    "elemental",
    "built",
    "people",
    "blur"
]

# TouchDesigner settings
TD_PROJECT_PATH = "/path/to/your/touchdesigner/project.toe"
TD_SCHEDULE_DAT = "scheduleDAT"

# Ensure directories exist
for directory in [INPUT_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Validation
def validate_config():
    """Validate that all required paths and settings are properly configured."""
    if not BASE_DIR.exists():
        raise ValueError(f"Base directory does not exist: {BASE_DIR}")
    
    if not all(isinstance(movement, str) for movement in MOVEMENTS):
        raise ValueError("All movements must be strings")
    
    if PRORES_PROFILE not in ["ProRes422", "ProRes422HQ", "ProRes4444"]:
        raise ValueError(f"Invalid ProRes profile: {PRORES_PROFILE}")
    
    if HAP_QUALITY not in ["low", "medium", "high"]:
        raise ValueError(f"Invalid HAP quality: {HAP_QUALITY}")

# Run validation on import
validate_config() 