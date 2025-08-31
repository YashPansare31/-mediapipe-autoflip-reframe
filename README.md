# Webinar Video Reframer

Automatically convert horizontal webinar videos to vertical 9:16 format for social media platforms (YouTube Shorts, Instagram Reels, TikTok).

## Features

- **Multi-method Face Detection**: MediaPipe + OpenCV fallbacks for robust speaker detection
- **Universal Layout Detection**: Works with any webinar layout (speaker left/right/corner/overlay)
- **Temporal Stabilization**: Smooth tracking prevents jitter and shaking
- **Quality Enhancement**: Sharpening for slides, denoising for faces
- **Edge Case Handling**: Robust processing even with poor video quality
- **Production Ready**: Optimized encoding, performance monitoring, error handling

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/webinar-reframer.git
cd webinar-reframer

# Install dependencies
pip install -r requirements.txt

# Optional: Install enhanced requirements for better quality
pip install -r requirements_enhanced.txt

```
### Basic Usage

# Simple reframing
python -m src.cli -i webinar.mp4 -o reframed.mp4

# High quality with preview
python -m src.cli -i input.mp4 -o output.mp4 --quality high --preview

# Custom settings
python -m src.cli -i video.mp4 -o result.mp4 --bitrate 6M --min-face-conf 0.2