"""
Validates the project directory structure and permissions.
Checks for required directories, files, and proper permissions.
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import re
from collections import defaultdict

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import config
from config import (
    PROJECT_ROOT,
    DATA_DIR,
    SOURCE_DIR,
    PRORES_DIR,
    PLAYBACK_DIR,
    SCENES_DIR
)

REQUIRED_TAGS = ["formal_tags", "emotional_tags", "material_tags", "semantic_tags"]
missing_tags = {key: [] for key in REQUIRED_TAGS}

def check_command(command):
    """
    Check if a command is available in the system.
    
    Args:
        command: Command to check
        
    Returns:
        Boolean indicating if command is available
    """
    try:
        subprocess.run(
            [command, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except FileNotFoundError:
        return False

def check_directory(path, required=True):
    """
    Check if a directory exists and is accessible.
    
    Args:
        path: Path to check (string or Path object)
        required: Whether this directory is required
        
    Returns:
        Boolean indicating if directory is valid
    """
    try:
        # Convert string to Path if needed
        path = Path(path) if isinstance(path, str) else path
        
        if not path.exists():
            if required:
                print(f"❌ Required directory missing: {path}")
                return False
            else:
                print(f"ℹ️ Optional directory missing: {path}")
                return True
                
        if not os.access(path, os.R_OK | os.W_OK):
            print(f"❌ Permission error: Cannot read/write to {path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking directory {path}: {str(e)}")
        return False

def check_file(path, required=True):
    """
    Check if a file exists and is accessible.
    
    Args:
        path: Path to check (string or Path object)
        required: Whether this file is required
        
    Returns:
        Boolean indicating if file is valid
    """
    try:
        # Convert string to Path if needed
        path = Path(path) if isinstance(path, str) else path
        
        if not path.exists():
            if required:
                print(f"❌ Required file missing: {path}")
                return False
            else:
                print(f"ℹ️ Optional file missing: {path}")
                return True
                
        if not os.access(path, os.R_OK):
            print(f"❌ Permission error: Cannot read {path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking file {path}: {str(e)}")
        return False

def validate_structure():
    """
    Validate the entire project structure.
    Checks all required directories and files.
    
    Returns:
        Boolean indicating if structure is valid
    """
    print("\nValidating runtime project structure...\n")
    
    # Check required tools
    print("Checking required tools...")
    tools = {
        "ffmpeg": "Video processing and conversion",
        "ffprobe": "Video metadata extraction"
    }
    
    for tool, purpose in tools.items():
        if not check_command(tool):
            print(f"❌ Required tool missing: {tool} ({purpose})")
            return False
        print(f"✅ Found {tool}")
    
    # Check project root
    if not check_directory(PROJECT_ROOT):
        return False
        
    # Check data directory
    if not check_directory(DATA_DIR):
        return False
        
    # Check video processing directories
    video_dirs = [
        SOURCE_DIR,
        PRORES_DIR,
        PLAYBACK_DIR,
        SCENES_DIR
    ]
    
    for directory in video_dirs:
        if not check_directory(directory):
            return False
            
    # Check configuration files
    config_files = [
        Path("requirements.txt"),
        Path(".env.example")
    ]
    
    for file in config_files:
        if not check_file(file):
            return False
            
    # Check JSON files for missing tag categories
    for filename in os.listdir(config.OUTPUT_DIR):
        if filename.endswith('.json') and not filename.startswith('._'):
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('metadata_payload', {})
                for key in REQUIRED_TAGS:
                    if not metadata.get(key):
                        missing_tags[key].append(filename)

    print("\nValidation Results: Missing tag categories:")
    for key, files in missing_tags.items():
        if files:
            print(f"{key}: {files}")
        else:
            print(f"{key}: None (all files have this tag)")

    print("\n✅ Project structure validation complete!")
    return True

def report_tag_inconsistencies():
    """
    Generate a report of tag inconsistencies across all JSON files.
    This includes plural/singular, case/character normalization, and potential synonyms.
    """
    print("\nGenerating tag inconsistency report...")
    tag_frequency = defaultdict(int)
    tag_variants = defaultdict(set)
    tag_categories = defaultdict(set)
    tag_files = defaultdict(list)

    for filename in os.listdir(config.OUTPUT_DIR):
        if filename.endswith('.json') and not filename.startswith('._'):
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('metadata_payload', {})
                for key in REQUIRED_TAGS:
                    if key in metadata:
                        for tag in metadata[key]:
                            tag_frequency[tag] += 1
                            tag_variants[tag.lower()].add(tag)
                            tag_categories[tag].add(key)
                            tag_files[tag].append(filename)

    print("\nTag Inconsistency Report:")
    print("------------------------")
    print("1. Plural/Singular Inconsistencies:")
    for tag, variants in tag_variants.items():
        if len(variants) > 1:
            print(f"  {tag}: {variants}")

    print("\n2. Case/Character Inconsistencies:")
    for tag, variants in tag_variants.items():
        if len(variants) > 1:
            print(f"  {tag}: {variants}")

    print("\n3. Tags with Special Characters or Numbers:")
    for tag in tag_frequency:
        if re.search(r'[^a-z_]', tag):
            print(f"  {tag}")

    print("\n4. Tags with Low Frequency (Potential Typos):")
    for tag, freq in tag_frequency.items():
        if freq < 2:
            print(f"  {tag} (frequency: {freq})")

    print("\n5. Tags with High Frequency (Potentially Generic):")
    for tag, freq in tag_frequency.items():
        if freq > 10:
            print(f"  {tag} (frequency: {freq})")

    print("\n6. Tags with Multiple Categories:")
    for tag, categories in tag_categories.items():
        if len(categories) > 1:
            print(f"  {tag}: {categories}")

    print("\n7. Tags with Multiple Files:")
    for tag, files in tag_files.items():
        if len(files) > 1:
            print(f"  {tag}: {files}")

    print("\nTag Inconsistency Report Complete!")

def main():
    """Main entry point for validation."""
    if not validate_structure():
        print("\n❌ Validation failed. Please fix the issues above.")
        sys.exit(1)
        
    print("\nProcessing workflow:")
    print("1. Copy your source videos to:", SOURCE_DIR)
    print("2. Run convert_to_prores.py to convert source videos to ProRes format")
    print("   - This will create ProRes files in:", PRORES_DIR)
    print("3. Run scripts/process_videos.py to split scenes and generate metadata")
    print("   - This will create scene files and metadata in:", SCENES_DIR)
    print("4. Run scripts/convert_to_hap.py to create HAP files for playback")
    print("   - This will create HAP files in:", PLAYBACK_DIR)
    print("\nRequired tools:")
    print("- ffmpeg: For video processing and conversion")
    print("- ffprobe: For video metadata extraction")
    print("\nNote: Each step must be completed in order as each script depends on the output of the previous step.")

    # Generate tag inconsistency report
    report_tag_inconsistencies()

if __name__ == "__main__":
    main() 