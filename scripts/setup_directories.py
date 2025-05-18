"""
Sets up the directory structure for the runtime project.
Creates all necessary directories and verifies permissions.
"""

import os
import sys
from pathlib import Path
import config

def create_directory_structure():
    """Create the directory structure for the project."""
    print("\nSetting up runtime project directories...\n")
    
    # Create main directories
    directories = [
        (config.DATA_DIR, "Data directory"),
        (config.SOURCE_DIR, "Source videos"),
        (config.PRORES_DIR, "ProRes converted videos"),
        (config.PLAYBACK_DIR, "HAP encoded videos"),
        (config.SCENES_DIR, "Scene-split videos"),
        (config.THUMBNAILS_DIR, "Video thumbnails"),
        (config.OUTPUT_DIR, "Processed descriptions")
    ]
    
    for directory, description in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory} ({description})")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {str(e)}")
            return False
    
    # Create .gitkeep files to preserve empty directories
    for directory, _ in directories:
        gitkeep_file = directory / ".gitkeep"
        try:
            gitkeep_file.touch()
            print(f"✅ Added .gitkeep to {directory}")
        except Exception as e:
            print(f"❌ Failed to create .gitkeep in {directory}: {str(e)}")
            return False
    
    print("\nDirectory structure setup complete!")
    return True

def main():
    """Main entry point for directory setup."""
    print("\nRuntime Project Directory Setup")
    print("==============================")
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Data directory: {config.DATA_DIR}")
    
    if not create_directory_structure():
        print("\n❌ Failed to set up directory structure.")
        sys.exit(1)
    
    print("\nNext steps:")
    print("1. Place your source video files in:", config.SOURCE_DIR)
    print("2. Run validate_structure.py to verify the setup")
    print("3. Begin processing videos with the pipeline scripts")
    print("\nNote: All data will be stored within the project directory")
    print("      No external volumes are required")

if __name__ == "__main__":
    main() 