"""
Validates the project directory structure and permissions.
Checks for required directories, files, and proper permissions.
"""

import os
import sys
import subprocess
from pathlib import Path
import config

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
    if not check_directory(config.PROJECT_ROOT):
        return False
        
    # Check data directory
    if not check_directory(config.DATA_DIR):
        return False
        
    # Check video processing directories
    video_dirs = [
        config.SOURCE_DIR,
        config.PRORES_DIR,
        config.PLAYBACK_DIR,
        config.SCENES_DIR
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
            
    print("\n✅ Project structure validation complete!")
    return True

def main():
    """Main entry point for validation."""
    if not validate_structure():
        print("\n❌ Validation failed. Please fix the issues above.")
        sys.exit(1)
        
    print("\nProcessing workflow:")
    print("1. Copy your source videos to:", config.SOURCE_DIR)
    print("2. Run convert_to_prores.py to convert source videos to ProRes format")
    print("   - This will create ProRes files in:", config.PRORES_DIR)
    print("3. Run scripts/process_videos.py to split scenes and generate metadata")
    print("   - This will create scene files and metadata in:", config.SCENES_DIR)
    print("4. Run scripts/convert_to_hap.py to create HAP files for playback")
    print("   - This will create HAP files in:", config.PLAYBACK_DIR)
    print("\nRequired tools:")
    print("- ffmpeg: For video processing and conversion")
    print("- ffprobe: For video metadata extraction")
    print("\nNote: Each step must be completed in order as each script depends on the output of the previous step.")

if __name__ == "__main__":
    main() 