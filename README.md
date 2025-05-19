# runtime

*"runtime is recognition through repetition — meaning built from echo."*

A live audio-visual performance exploring the loop between memory and motion. Part of the IN.SIGHT PERFORMANCE SERIES, a collaboration between [Groupwork](https://groupwork.fyi) and [Torn Space Theater](https://tornspacetheater.com).

## Overview

runtime is a generative performance system that transforms personal digital archives into a living, breathing composition. The initial version was built from over two thousand video fragments spanning twenty-five years of digital life. The system processes, analyzes, and recombines these memories into a dynamic performance that moves through five distinct movements: orientation, elemental, built, people, and blur.

This project is part of the IN.SIGHT PERFORMANCE SERIES, exploring the intersection of memory and motion through digital archives. It leverages advanced video processing techniques and AI-driven metadata generation to create a dynamic, generative performance. The system is designed to be flexible and extensible, allowing for the integration of new content and performance elements as needed. The collaboration with Torn Space Theater highlights the project's commitment to innovative, interdisciplinary art forms.

## Dependencies and Requirements

- **Python**: Ensure you have Python 3.8 or higher installed.
- **FFmpeg**: Required for video processing tasks. Install it via your package manager or from [ffmpeg.org](https://ffmpeg.org/download.html).
- **TouchDesigner**: Used for real-time video playback. Download from [derivative.ca](https://derivative.ca/download).
- **Ableton Live**: Required for audio processing and performance. The project uses Ableton Live 11 or higher.
- **BlackHole**: Virtual audio driver for routing audio between applications. Download from [existential.audio](https://existential.audio/blackhole/).
- **Python Packages**: Install the required Python packages using the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

- **scripts/**: Contains all pre-processing scripts used in the video processing pipeline. These scripts handle tasks such as video conversion, scene detection, metadata generation, and performance scheduling, forming the backbone of the runtime system's data preparation and organization.
- **data/**: Contains source video files and generated metadata.
- **movement_csvs/**: Stores CSV files for each movement phase.
- **runtime-env/**: Environment-specific configurations and scripts.
- **config.py**: Configuration settings for paths and processing options.
- **requirements.txt**: Lists Python package dependencies.
- **archived-performance-schedules/**: Contains past performance schedules.
- **ableton/**: Contains the Ableton Live project and Max for Live patches for the generative soundtrack.

## Technical Architecture

The system consists of several interconnected components:

### Video Processing Pipeline
1. **Initial Processing**
   - `scripts/convert_to_prores.py`: Converts source videos to ProRes format with parallel processing, error handling, and audio normalization
   - `scripts/split_scenes.py`: Analyzes and splits videos into distinct scenes
   - `scripts/process_videos.py`: Processes video for analysis using BLIP-2, generates metadata

2. **Metadata Generation**
   - `scripts/check_captions.py`: Validates and filters generated captions, identifies and regenerates poor quality captions
   - `scripts/generate_descriptions.py`: Uses GPT-4 to generate rich descriptions
   - `scripts/resolve_conflicts.py`: Resolves semantic conflicts in video tags
   - `scripts/normalize_tags.py`: Standardizes metadata tags across the collection
   - `scripts/generate_embeddings.py`: Creates semantic embeddings for content-based retrieval
   - `scripts/find_similar_videos.py`: Debugging tool to verify embedding quality and metadata consistency
     - Uses semantic search to find similar videos
     - Helps identify potential issues with descriptions or tags
     - Minimum similarity threshold of 0.3 to ensure quality matches
     - Useful for verifying the semantic coherence of the metadata

3. **Performance Preparation**
   - `scripts/convert_to_hap.py`: Creates HAP-encoded versions for TouchDesigner playback
   - `scripts/update_json_paths.py`: Updates JSON metadata to reference HAP-encoded files
   - `scripts/assign_phases.py`: Assigns videos to performance phases
   - `scripts/generate_movement_csv.py`: Creates individual CSV files for each movement phase (orientation.csv, elemental.csv, etc.)
   - `scripts/emit_playback.py`: Combines movement CSVs into a single performance_schedule.csv for TouchDesigner playback control
   - `scripts/dat_chopexec.py`: Provides the core playback control logic used in the TouchDesigner DAT ChopExec component, managing video transitions, timing, and layer composition during performance

### TouchDesigner Integration
The system uses TouchDesigner for real-time video playback and composition. The `dat_chopexec.py` script is a critical component that:
- Manages video transitions and timing
- Controls layer composition and blending
- Handles performance phase transitions
- Coordinates with the movement schedule
- Provides real-time control over playback parameters

This script is integrated into the TouchDesigner project through a DAT ChopExec component, which executes the Python code for precise control over video playback during performance. The TouchDesigner file (`runtime.toe`) reads the `performance_schedule.csv` file generated by `emit_playback.py` to determine the sequence and timing of video playback, ensuring synchronized performance across all movements.

## Setup

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv runtime
   source runtime/bin/activate  # or `runtime\Scripts\activate` on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Copy `.env.example` to `.env` and add your OpenAI API key
   - Configure `scripts/config.py` with your system paths and settings:
     ```python
      # Base directories
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
      ```   

3. **Processing Pipeline**
   ```bash
   # 1. Initial Processing
   python scripts/convert_to_prores.py  # Converts source videos to ProRes format
   
   # Check conversion_errors.txt for any failed conversions
   # Review _SKIPPED_OR_CORRUPTED directory for problematic files
   # Once all videos are successfully converted to ProRes, proceed with:
   
   python scripts/split_scenes.py
   python scripts/process_videos.py
   
   # 2. Metadata Generation
   python scripts/check_captions.py
   python scripts/generate_descriptions.py
   python scripts/resolve_conflicts.py
   python scripts/normalize_tags.py
   python scripts/generate_embeddings.py
   python scripts/find_similar_videos.py
   
   # 3. Performance Preparation
   python scripts/convert_to_hap.py
   python scripts/update_json_paths.py
   python scripts/assign_phases.py
   python scripts/generate_movement_csv.py
   python scripts/emit_playback.py
   ```

## Performance Structure

The performance unfolds through five movements:

1. **Orientation**: Introduction to the system and its capabilities
2. **Elemental**: Exploration of basic visual elements and patterns
3. **Built**: Examination of constructed environments and spaces
4. **People**: Focus on human presence and interaction
5. **Blur**: Dissolution of boundaries between elements

Each movement is characterized by:
- Distinct visual language
- Specific clip selection criteria
- Unique layer composition rules
- Movement-specific audio processing

## Development

### Processing Pipeline

The development workflow follows these key steps:

1. **Initial Processing**
   - Convert source videos to ProRes format
   - Split videos into scene segments
   - Process videos for metadata extraction

2. **Metadata Generation**
   - Validate and filter generated captions
   - Regenerate poor quality captions
   - Generate AI-powered descriptions and tags using GPT-4
   - Resolve semantic conflicts in tags
   - Normalize metadata tags
   - Create semantic embeddings

3. **Performance Preparation**
   - Convert ProRes scenes to HAP codec
   - Update JSON file paths for HAP videos
   - Assign videos to performance phases
   - Generate movement scheduling
   - Finalize performance configuration
   - Test and refine playback

### Key Components

- **Video Processing**: FFmpeg-based conversion and analysis with parallel processing and error handling
- **AI Integration**: OpenAI GPT-4 and BLIP-2 for metadata generation
- **Performance Engine**: TouchDesigner-based real-time composition
- **Data Management**: JSON-based metadata storage and retrieval

### Adding New Content

#### 1. Preparing Source Videos

- **Gather your video files**
  - Ensure videos are complete and not corrupted
  - Remove any temporary or partial exports
  - Consider trimming very long videos into more manageable segments

- **Organize source material**
  - Place all video files in the `data/source` directory
  - Use descriptive filenames without special characters
  - Supported formats: `.mov`, `.mp4`, `.avi`, `.mkv`, `.dv` (case-insensitive)

- **Check video properties**
  - Ensure videos have sufficient visual content (avoid mostly black/static videos)
  - For optimal results, aim for videos between 10 seconds and 10 minutes
  - If using DV files, note they will be automatically deinterlaced and scaled

#### 2. Running the Processing Pipeline

- **Initial conversion to ProRes**
  ```bash
  python scripts/convert_to_prores.py
  ```
  - This converts all source videos to ProRes format for consistent processing
  - Handles various input formats and normalizes them
  - Creates `data/prores` directory with converted files

- **Verify conversion success**
  ```bash
  # Check for conversion errors
  cat data/conversion_errors.txt  # if it exists
  
  # Examine any problematic files
  ls data/_SKIPPED_OR_CORRUPTED
  ```
  - Address any conversion issues before proceeding
  - Common issues include corrupt files, permission problems, or unsupported codecs

- **Split videos into scene segments**
  ```bash
  python scripts/split_scenes.py
  ```
  - Analyzes video content and splits at natural scene transitions
  - Creates individual scene files in `data/scenes` directory
  - Generates initial scene detection metadata

- **Process videos for metadata extraction**
  ```bash
  python scripts/process_videos.py
  ```
  - Analyzes each scene for visual content, color, and motion
  - Uses BLIP-2 AI model for initial frame captioning
  - Generates thumbnails for each scene
  - Creates initial JSON metadata for each scene in `data/descriptions`

#### 3. Metadata Enhancement and Validation

- **Validate generated captions**
  ```bash
  python scripts/check_captions.py
  ```
  - Identifies poor quality or problematic captions
  - Flags videos that need improved descriptions
  - Creates a list of scenes requiring regeneration
  - Uses alternative prompts or settings for previously flagged scenes
  - Improves caption quality through targeted refinement
  - Updates metadata with improved descriptions

- **Generate AI-powered descriptions and tags**
  ```bash
  python scripts/generate_descriptions.py
  ```
  - Uses GPT-4 to enhance descriptions created by BLIP-2
  - Identifies key visual elements and themes
  - Enhances metadata with semantic understanding of content

- **Resolve semantic conflicts**
  ```bash
  python scripts/resolve_conflicts.py
  ```
  - Harmonizes contradictory tags or descriptions
  - Improves consistency across the collection
  - Creates a more coherent semantic structure

- **Normalize metadata tags**
  ```bash
  python scripts/normalize_tags.py
  ```
  - Standardizes terminology across tags
  - Merges similar concepts for consistency
  - Improves searchability and organization

- **Generate semantic embeddings**
  ```bash
  python scripts/generate_embeddings.py
  ```
  - Creates vector representations of video content
  - Enables semantic searching and grouping
  - Prepares for phase assignment

- **Find similar videos**
  ```bash
  python scripts/find_similar_videos.py
  ```
  - Uses semantic search to find similar videos
  - Helps identify potential issues with descriptions or tags
  - Minimum similarity threshold of 0.3 to ensure quality matches
  - Useful for verifying the semantic coherence of the metadata

#### 4. Performance Integration

- **Create optimized playback formats**
  ```bash
  python scripts/convert_to_hap.py
  ```
  - Converts ProRes scenes to HAP codec for efficient playback
  - Optimizes for TouchDesigner performance
  - Creates files in `data/playback` directory 

- **Update JSON file paths for HAP videos**
  ```bash
  python scripts/update_json_paths.py
  ```
  - Copies and updates JSON metadata files to point to HAP video files
  - Ensures TouchDesigner uses the optimized video format
  - Updates file paths in metadata to reference the HAP-encoded versions

- **Assign videos to performance phases**
  ```bash
  python scripts/assign_phases.py
  ```
  - Analyzes content and assigns to appropriate movements
  - Considers visual themes, motion, and content
  - Creates phase assignments for each video segment

- **Generate movement scheduling**
  ```bash
  python scripts/generate_movement_csv.py
  ```
  - Creates CSV files for each movement phase
  - Determines optimal sequencing within each movement
  - Defines playback order and timing hints

- **Finalize performance configuration**
  ```bash
  python scripts/emit_playback.py
  ```
  - Generates the final playback configuration
  - Sets up all parameters needed for the TouchDesigner environment
  - Creates the complete performance schedule

#### 5. Testing and Refinement

- **Load the TouchDesigner environment**
  - Open `runtime.toe` in TouchDesigner
  - Ensure all directories are correctly referenced
  - Check that video files are properly detected

- **Test playback in each movement**
  - Preview scenes from each movement phase
  - Verify transitions between videos
  - Check for any performance issues or glitches

- **Make refinements as needed**
  - Adjust phase assignments for misclassified videos
  - Re-process problematic videos
  - Fine-tune movement transitions

- **Backup your work**
  - Create a backup of your final configuration
  - Archive valuable metadata for future reference
  - Consider versioning important configuration files

## Audio System

The performance uses a multi-layered audio system that combines generative composition with real-time processing:

### Components
- **Ableton Live Project**: Contains the main composition structure and mixer setup
- **Max for Live Patches**: Custom patches for generative audio processing and control
- **BlackHole Audio Routing**: [BlackHole](https://existential.audio/blackhole/) virtual audio driver for routing audio between applications

### Setup and Configuration
1. **Install and Configure BlackHole**
   - Download and install from [existential.audio](https://existential.audio/blackhole/)
   - Set up as audio output in TouchDesigner and input in Ableton Live
   - Route video layer audio through separate BlackHole channels

2. **Ableton Live Setup**
   - Open `ableton/Runtime Project/runtime.als`
   - Replace proprietary VST instruments with your preferred synths:
     - Ambient pad textures
     - Granular processing
     - Spectral manipulation
     - Movement-based sound design
   - Configure mixer channels for each video layer
   - Load and configure Max for Live patches for:
     - Video metadata response
     - Movement-based parameter control
     - Real-time audio processing

### Performance
- Each movement has its own audio processing chain
- Video metadata influences generative parameters
- Real-time mixing and effects processing
- Synchronized audio-visual transitions between movements

## Credits

- Created by Frank Napolski
- Part of the IN.SIGHT PERFORMANCE SERIES
- Presented by [Groupwork](https://groupwork.fyi) and [Torn Space Theater](https://tornspacetheater.com)
- Supported by the Cullen Foundation and the National Endowment for the Arts

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

This means you are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- NonCommercial — You may not use the material for commercial purposes
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

For more information, see the [LICENSE](LICENSE) file.
