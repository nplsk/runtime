# runtime

*"runtime is recognition through repetition — meaning built from echo."*

A live audio-visual performance exploring the loop between memory and motion. Part of the IN.SIGHT PERFORMANCE SERIES, a collaboration between [Groupwork](https://groupwork.fyi) and [Torn Space Theater](https://tornspacetheater.com).

## Overview

runtime is a generative performance system that transforms personal digital archives into a living, breathing composition. Built from over two thousand video fragments spanning twenty-five years of digital life, the system processes, analyzes, and recombines these memories into a dynamic performance that moves through five distinct movements: orientation, elemental, built, people, and blur.

## Technical Architecture

The system consists of several interconnected components:

### Video Processing Pipeline
1. **Initial Processing**
   - `scripts/convert_to_prores.py`: Converts source videos to ProRes format with parallel processing, error handling, and audio normalization
   - `scripts/split_scenes.py`: Analyzes and splits videos into distinct scenes
   - `scripts/process_videos.py`: Processes video metadata and prepares for analysis

2. **Format Conversion**
   - `scripts/convert_to_hap.py`: Creates HAP-encoded versions for TouchDesigner playback
   - `scripts/update_json_paths.py`: Updates JSON metadata to reference HAP-encoded files

3. **Metadata Generation**
   - `scripts/generate_descriptions.py`: Uses BLIP-2 and GPT-4 to generate rich descriptions
   - `scripts/generate_embeddings.py`: Creates semantic embeddings for content-based retrieval
   - `scripts/check_captions.py`: Validates and filters generated captions
   - `scripts/regenerate_bad_captions.py`: Identifies and regenerates poor quality captions

4. **Content Organization**
   - `scripts/resolve_conflicts.py`: Resolves semantic conflicts in video tags
   - `scripts/normalize_tags.py`: Standardizes metadata tags across the collection
   - `scripts/assign_phases.py`: Assigns videos to performance phases
   - `scripts/generate_movement_csv.py`: Creates the performance schedule

5. **Performance Engine**
   - `scripts/emit_playback.py`: Generates the final playback configuration
   - TouchDesigner-based playback system (`runtime.toe`)
   - Multi-layer video composition
   - Real-time synchronization with audio
   - Movement-based clip scheduling

## Setup

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Copy `.env.example` to `.env` and add your OpenAI API key
   - Configure `config.py` with your system paths and settings:
     ```python
     # Base directories
     BASE_DIR = Path("/path/to/your/runtime/directory")
     PROCESSED_DIR = BASE_DIR / "PROCESSED"
     PROCESSED_DESCRIPTIONS_DIR = BASE_DIR / "PROCESSED_DESCRIPTIONS"
     
     # Video processing settings
     PRORES_PROFILE = "ProRes422"  # Options: ProRes422, ProRes422HQ, ProRes4444
     HAP_QUALITY = "high"  # Options: low, medium, high
     
     # AI settings
     BLIP_MODEL = "Salesforce/blip2-opt-2.7b"
     EMBEDDING_MODEL = "all-MiniLM-L6-v2"
     ```
   - Update the TouchDesigner project path in `config.py` to point to your `runtime.toe` file

3. **Processing Pipeline**
   ```bash
   # 1. Initial Processing
   python scripts/convert_to_prores.py  # Converts source videos to ProRes format
   
   # Check conversion_errors.txt for any failed conversions
   # Review _SKIPPED_OR_CORRUPTED directory for problematic files
   # Once all videos are successfully converted to ProRes, proceed with:
   
   python scripts/split_scenes.py
   python scripts/process_videos.py
   
   # 2. Format Conversion
   python scripts/convert_to_hap.py
   python scripts/update_json_paths.py
   
   # 3. Metadata Generation
   python scripts/generate_descriptions.py
   python scripts/generate_embeddings.py
   python scripts/check_captions.py
   python scripts/regenerate_bad_captions.py
   
   # 4. Content Organization
   python scripts/resolve_conflicts.py
   python scripts/normalize_tags.py
   python scripts/assign_phases.py
   python scripts/generate_movement_csv.py
   
   # 5. Performance Setup
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

- **Generate AI-powered descriptions and tags**
  ```bash
  python scripts/generate_descriptions.py
  ```
  - Uses GPT-4 to enhance descriptions created by BLIP-2
  - Identifies key visual elements and themes
  - Enhances metadata with semantic understanding of content

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

#### 3. Metadata Enhancement and Validation

- **Generate semantic embeddings**
  ```bash
  python scripts/generate_embeddings.py
  ```
  - Creates vector representations of video content
  - Enables semantic searching and grouping
  - Prepares for phase assignment

- **Validate generated captions**
  ```bash
  python scripts/check_captions.py
  ```
  - Identifies poor quality or problematic captions
  - Flags videos that need improved descriptions
  - Creates a list of scenes requiring regeneration

- **Regenerate problematic captions**
  ```bash
  python scripts/regenerate_bad_captions.py
  ```
  - Uses alternative prompts or settings for previously flagged scenes
  - Improves caption quality through targeted refinement
  - Updates metadata with improved descriptions

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

#### 4. Performance Integration

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

## Dependencies and Requirements

- **Python**: Ensure you have Python 3.8 or higher installed.
- **FFmpeg**: Required for video processing tasks. Install it via your package manager or from [ffmpeg.org](https://ffmpeg.org/download.html).
- **TouchDesigner**: Used for real-time video playback. Download from [derivative.ca](https://derivative.ca/download).
- **Python Packages**: Install the required Python packages using the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

## Detailed Usage Instructions

1. **Initial Setup**
   - Follow the environment setup instructions to create and activate a virtual environment.
   - Install all necessary dependencies as outlined above.

2. **Running the Processing Pipeline**
   - Start with the ProRes conversion:
     ```bash
     python scripts/convert_to_prores.py
     ```
   - Follow the sequence of scripts as described in the Processing Pipeline section to complete the setup.

3. **Playback and Performance**
   - Use TouchDesigner to load the `runtime.toe` file for real-time playback.
   - Ensure all video files are correctly processed and available in the specified directories.

## Project Structure

- **scripts/**: Contains all pre-processing scripts used in the video processing pipeline. These scripts handle tasks such as video conversion, scene detection, metadata generation, and performance scheduling, forming the backbone of the runtime system's data preparation and organization.
- **data/**: Contains source video files and generated metadata.
- **movement_csvs/**: Stores CSV files for each movement phase.
- **runtime-env/**: Environment-specific configurations and scripts.
- **config.py**: Configuration settings for paths and processing options.
- **requirements.txt**: Lists Python package dependencies.
- **archived-performance-schedules/**: Contains past performance schedules.

## Additional Context or Background

This project is part of the IN.SIGHT PERFORMANCE SERIES, exploring the intersection of memory and motion through digital archives. It leverages advanced video processing techniques and AI-driven metadata generation to create a dynamic, generative performance. The system is designed to be flexible and extensible, allowing for the integration of new content and performance elements as needed. The collaboration with Torn Space Theater highlights the project's commitment to innovative, interdisciplinary art forms.
