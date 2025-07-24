# MMF to MP4 Converter

Convert MMF (Media Memory File) files to MP4 videos using the command line.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg:
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

## Usage

Basic conversion:
```bash
python mmf_decoder.py input.mmf output.mp4
```

With custom settings:
```bash
python mmf_decoder.py input.mmf output.mp4 --fps 30 --quality 23
```

### Options

- `--fps`: Frame rate (default: 20)
- `--quality`: CRF quality 0-51, lower = better quality (default: 28)
- `--max-frames`: Maximum frames to convert (for testing)

### Examples

```bash
# Convert with default settings
python mmf_decoder.py video.mmf video.mp4

# High quality conversion
python mmf_decoder.py video.mmf video.mp4 --quality 18

# Test with first 100 frames
python mmf_decoder.py video.mmf test.mp4 --max-frames 100

# Slow motion (lower FPS)
python mmf_decoder.py video.mmf slow.mp4 --fps 10
```

## Quality Settings

- **18-23**: High quality (larger files)
- **24-28**: Good quality (recommended)
- **29-35**: Lower quality (smaller files)

## Requirements

- Python 3.7+
- FFmpeg
- Dependencies listed in requirements.txt

