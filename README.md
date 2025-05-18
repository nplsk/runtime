# runtime

A live audio-visual performance exploring the loop between memory and motion. Part of the IN.SIGHT PERFORMANCE SERIES in collaboration with Torn Space Theater.

## Overview

runtime is a generative performance system that transforms personal digital archives into a living, breathing composition. Built from over two thousand video fragments spanning twenty-five years of digital life, the system processes, analyzes, and recombines these memories into a dynamic performance that moves through five distinct movements: orientation, elemental, built, people, and blur.

## Technical Architecture

The system consists of several interconnected components:

### Video Processing Pipeline
1. **Initial Processing**
   - `convert_all_to_prores.py`: Converts video files to ProRes format for optimal playback
   - `split_scenes.py`: Analyzes and splits videos into distinct scenes
   - `process_videos.py`: Processes video metadata and prepares for analysis

2. **Format Conversion**
   - `convert_to_hap.py`: Creates HAP-encoded versions for TouchDesigner playback

3. **Metadata Generation**
   - `generate_descriptions.py`: Uses BLIP-2 and GPT-4 to generate rich descriptions
   - `generate_embeddings.py`: Creates semantic embeddings for content-based retrieval
   - `check_captions.py`: Validates and filters generated captions
   - `regenerate_bad_captions.py`: Identifies and regenerates poor quality captions

4. **Content Organization**
   - `resolve_conflicts.py`: Resolves semantic conflicts in video tags
   - `normalize_tags.py`: Standardizes metadata tags across the collection
   - `assign_phases.py`: Assigns videos to performance phases
   - `generate_movement_csv.py`: Creates the performance schedule

5. **Performance Engine**
   - `emit_playback.py`: Generates the final playback configuration
   - TouchDesigner-based playback system with multi-layer composition

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

3. **Processing Pipeline**
   ```bash
   # 1. Initial Processing
   python convert_all_to_prores.py
   python split_scenes.py
   python process_videos.py
   
   # 2. Format Conversion
   python convert_to_hap.py
   
   # 3. Metadata Generation
   python generate_descriptions.py
   python generate_embeddings.py
   python check_captions.py
   python regenerate_bad_captions.py
   
   # 4. Content Organization
   python resolve_conflicts.py
   python normalize_tags.py
   python assign_phases.py
   python generate_movement_csv.py
   
   # 5. Performance Setup
   python emit_playback.py
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

- **Video Processing**: FFmpeg-based conversion and analysis
- **AI Integration**: OpenAI GPT-4 and BLIP-2 for metadata generation
- **Performance Engine**: TouchDesigner-based real-time composition
- **Data Management**: JSON-based metadata storage and retrieval

### Adding New Content

1. Place new video files in the input directory
2. Run the processing pipeline in sequence
3. Review and adjust generated metadata
4. Update the performance schedule

## Credits

- Created by Frank Napolski
- Part of the IN.SIGHT PERFORMANCE SERIES
- Presented by Groupwork and Torn Space
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

---

*"runtime is recognition through repetition — meaning built from echo."*
