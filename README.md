# Webinar Video Reframer

Automatically convert horizontal webinar videos to vertical 9:16 format for social media platforms (YouTube Shorts, Instagram Reels, TikTok).

## Features

- **Intelligent Face Detection**: Multi-method face detection using MediaPipe Face Detection, Face Mesh, and OpenCV Haar Cascades
- **Universal Layout Recognition**: Automatically adapts to any webinar layout (speaker left/right/corner/overlay)
- **Temporal Stabilization**: Advanced smoothing prevents jitter and maintains stable framing
- **Quality Enhancement**: Automatic brightness correction, sharpening for slides, and noise reduction
- **Audio Preservation**: Maintains original audio quality in output video
- **Edge Case Handling**: Robust processing handles poor lighting, movement, and detection failures
- **Production Ready**: Optimized encoding, performance monitoring, configurable quality presets

## Quick Start

### Installation
#### Non-Docker Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd webinar-reframer

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify FFmpeg is installed (required for audio)
ffmpeg -version
```
### Docker 
#### Prerequisites

 Docker Desktop installed and running
 Git (to clone this repository)

### Build
```bash
docker build -t mp-autoflip . -f docker/Dockerfile
```
### Run
```bash
docker run --rm -v $PWD:/work mp-autoflip python -m src.reframe \
    --in data/input.mp4 \
    --out out/output_916.mp4 \
    --config configs/portrait_916.pbtxt
```

### Using Docker Compose (Alternative)
```bash
# Build
docker-compose build

# Run interactively
docker-compose run --rm mp-autoflip bash

# Inside container
python -m src.reframe --in data/input.mp4 --out out/output_916.mp4 --config configs/
### Basic Usage
```

## Basic Usage
```bash
# Simple reframing
python -m src.cli -i webinar.mp4 -o reframed.mp4

# High quality processing
python -m src.cli -i input.mp4 -o output.mp4 --quality high

# Fast preview (30 seconds)
python -m src.cli -i video.mp4 -o preview.mp4 --quality fast --preview

# Custom settings
python -m src.cli -i video.mp4 -o result.mp4 --bitrate 6M --min-face-conf 0.2
```

## Requirements

### System Requirements
- Python 3.10 or higher
- FFmpeg (for audio processing)
- 4GB+ RAM recommended
- Windows, macOS, or Linux

### Python Dependencies
```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
protobuf>=3.20.0
```

### FFmpeg Installation

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Configuration Options

### Quality Presets

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `fast` | ~4x real-time | Good | Quick previews, testing |
| `balanced` | ~2x real-time | High | Most use cases (default) |
| `high` | ~1x real-time | Maximum | Final production output |

### Advanced Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--min-face-conf` | 0.3 | 0.1-1.0 | Face detection confidence threshold |
| `--face-tracking` | 0.3 | 0.1-0.7 | Face tracking smoothness (lower = more stable) |
| `--bitrate` | 8M | 4M-12M | Output video bitrate |

### Fine-tuning for Stability

For maximum stability (slower response):
- `--face-tracking 0.2` (more stable face tracking)
- `--min-face-conf 0.2` (detect faces more reliably)

For responsiveness (faster adaptation):
- `--face-tracking 0.4` (more responsive tracking)
- `--min-face-conf 0.4` (higher confidence threshold)

## Python API

### Basic Usage

```python
from src.production_pipeline import ProductionPipeline

# Configure pipeline
config = {
    'min_face_conf': 0.3,
    'face_alpha': 0.25,
    'slide_window': 12,
    'target_bitrate': '8M',
    'enable_quality_enhancement': True
}

# Process video
pipeline = ProductionPipeline(config)
success = pipeline.process_video_production("input.mp4", "output.mp4")

if success:
    print("Video processed successfully!")
```

### With Progress Monitoring

```python
def progress_callback(percent, current_frame, total_frames):
    print(f"Progress: {percent:.1f}% ({current_frame}/{total_frames})")

success = pipeline.process_video_production(
    "input.mp4", 
    "output.mp4", 
    progress_callback
)
```

## Architecture

The system uses a modular pipeline architecture:

```
Input Video
    ↓
Face Detection (MediaPipe + OpenCV)
    ↓
Region Detection (Universal Layout)
    ↓
Temporal Smoothing (EMA + Median Filtering)
    ↓
Conflict Resolution
    ↓
Layout Composition (9:16 Format)
    ↓
Quality Enhancement
    ↓
Video Encoding + Audio Merging
    ↓
Output Video (9:16 with Audio)
```

## Troubleshooting

### Common Issues

**No face detected:**
```bash
# Try lower confidence threshold
python -m src.cli -i video.mp4 -o output.mp4 --min-face-conf 0.2

# For very poor lighting
python -m src.cli -i video.mp4 -o output.mp4 --min-face-conf 0.1
```

**Faces are too dark in output:**
- The system automatically detects and brightens dark faces
- For very dark videos, try the `--quality high` preset

**Jittery/unstable tracking:**
```bash
# More stable tracking
python -m src.cli -i video.mp4 -o output.mp4 --face-tracking 0.2
```

**No audio in output:**
- Ensure FFmpeg is installed and accessible
- Check that input video has audio: `ffprobe -v quiet -show_streams input.mp4`

**Processing too slow:**
```bash
# Use fast preset
python -m src.cli -i video.mp4 -o output.mp4 --quality fast

# Or lower bitrate
python -m src.cli -i video.mp4 -o output.mp4 --bitrate 4M
```

**Large output file size:**
```bash
# Reduce bitrate
python -m src.cli -i video.mp4 -o output.mp4 --bitrate 4M

# Use fast preset
python -m src.cli -i video.mp4 -o output.mp4 --quality fast --bitrate 6M
```

### Debug Mode

For troubleshooting, use verbose output:
```bash
python -m src.cli -i video.mp4 -o output.mp4 --verbose
```

This shows:
- Detection success rates
- Processing times for each component
- Memory usage statistics
- Quality enhancement decisions

## Performance

### Typical Processing Speed
- **Fast preset**: 3-6x real-time
- **Balanced preset**: 2-4x real-time  
- **High preset**: 1-2x real-time

### System Requirements
- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB RAM, quad-core CPU
- **Memory usage**: 200-600MB depending on input resolution

### Optimization Tips

1. **For faster processing:**
   - Use `--quality fast`
   - Lower `--bitrate 4M`
   - Add `--preview` for testing

2. **For better quality:**
   - Use `--quality high`
   - Increase `--bitrate 10M`
   - Ensure good lighting in source video

3. **For stability:**
   - Lower `--face-tracking 0.2`
   - Use `--min-face-conf 0.2`

## Input Video Requirements

### Supported Formats
- **Containers**: MP4, AVI, MOV, MKV
- **Video codecs**: H.264, H.265, VP9
- **Audio codecs**: AAC, MP3, PCM
- **Resolution**: 720p minimum, 1080p+ recommended

### Optimal Input Characteristics
- Clear speaker visibility (facing camera)
- Good contrast between speaker and background
- Stable lighting conditions
- Clear slide/presentation content
- Speaker and slides clearly separated

### Webinar Layout Compatibility
The system works with any webinar layout:
- Side-by-side (speaker + slides)
- Picture-in-picture (speaker overlay)
- Corner webcam layouts
- Full-screen with speaker popup
- Screen sharing with webcam

## Output Specifications

- **Resolution**: 1080x1920 (9:16 aspect ratio)
- **Frame rate**: Matches input (typically 30fps)
- **Video codec**: H.264 (MP4)
- **Audio codec**: AAC or original (copied)
- **Bitrate**: 4-8 Mbps (configurable)
- **Compatibility**: Optimized for social media platforms

## Project Structure

```
webinar-reframer/
├── src/
│   ├── cli.py                 # Command-line interface
│   ├── production_pipeline.py # Main processing pipeline
│   ├── detectors/            # Face and region detection
│   │   ├── face_mp.py        # Multi-method face detection
│   │   └── zone_detector.py  # Universal layout detection
│   ├── trackers/             # Temporal smoothing
│   │   └── temporal.py       # Tracking and stabilization
│   ├── layout/               # Video composition
│   │   └── composer.py       # Frame layout and enhancement
│   ├── encoding/             # Video encoding
│   │   └── video_encoder.py  # FFmpeg integration
│   └── optimization/         # Performance optimization
│       └── performance.py    # Monitoring and optimization
├── configs/                  # Configuration files
├── data/samples/             # Test videos (if any)
├── output/                   # Generated output videos
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Examples

### Command Line Examples

```bash
# Basic conversion
python -m src.cli -i lecture.mp4 -o vertical_lecture.mp4

# High quality with custom settings
python -m src.cli -i webinar.mp4 -o result.mp4 \
  --quality high \
  --bitrate 8M \
  --min-face-conf 0.2

# Fast preview for testing
python -m src.cli -i long_webinar.mp4 -o preview.mp4 \
  --quality fast \
  --preview \
  --bitrate 4M

# Troubleshooting mode
python -m src.cli -i problematic.mp4 -o output.mp4 \
  --verbose \
  --min-face-conf 0.1 \
  --face-tracking 0.2
```

### Python API Examples

```python
# Basic processing
from src.production_pipeline import ProductionPipeline

pipeline = ProductionPipeline()
success = pipeline.process_video_production("input.mp4", "output.mp4")

# Custom configuration
config = {
    'min_face_conf': 0.25,
    'face_alpha': 0.2,
    'slide_window': 15,
    'target_bitrate': '6M'
}

pipeline = ProductionPipeline(config)
success = pipeline.process_video_production("input.mp4", "output.mp4")

# With progress monitoring
def show_progress(percent, current, total):
    print(f"Processing: {percent:.1f}% complete")

success = pipeline.process_video_production(
    "input.mp4", "output.mp4", show_progress
)
```

## Technical Details

### Detection Methods

**Face Detection (3-tier fallback):**
1. MediaPipe Face Detection (primary)
2. MediaPipe Face Mesh (backup)
3. OpenCV Haar Cascade (fallback)

**Region Detection:**
- Universal zone-based detection
- Adaptive to any webinar layout
- Conflict resolution between face and slide regions

### Temporal Smoothing

**Face Tracking:**
- Exponential Moving Average (EMA) with adaptive alpha
- Movement limiting to prevent jumps
- Confidence-based persistence during brief detection losses

**Slide Tracking:**
- Temporal median filtering
- Inertia-based stabilization
- Shot change detection and reset

### Quality Enhancement

**Face Region:**
- Automatic exposure correction for dark faces
- Noise reduction optimized for video calls
- Gentle sharpening to improve clarity

**Slide Region:**
- Text-optimized sharpening
- Contrast enhancement for readability
- Aspect ratio preservation

## Limitations

- Requires visible speaker face (profile views may not work well)
- Works best with clear speaker/slide separation
- Processing time scales with input video length
- Output quality depends on input video quality
- Extreme lighting conditions may affect detection

## License

MIT License - See LICENSE file for details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run with `--verbose` flag for detailed diagnostics
3. Ensure all requirements are properly installed
4. Verify input video meets the recommended specifications

## Version

Version 1.0 - Production Release
