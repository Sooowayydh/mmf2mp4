# MMF to MP4 Converter

Convert MMF (Media Memory File) files to MP4 videos easily. This tool is designed for users who need to convert their MMF media data to standard video format.

## 🚀 Quick Start (3 Steps)

### Step 1: Download this tool
Clone or download this repository to your computer.

### Step 2: Install requirements  
Open a terminal/command prompt in this folder and run:
```bash
python install.py
```
This will automatically install all needed packages and check your system.

### Step 3: Convert your files
**Option A - Double-click method (Easiest):**
- **Windows**: Double-click `convert.bat`
- **Mac/Linux**: Double-click `convert.sh` (or run `./convert.sh` in terminal)

**Option B - Command line:**
```bash
python mmf_decoder.py
```
The program will ask you questions and guide you through the conversion process.

## 📖 Detailed Instructions

### Installation

#### Option A: Automatic Installation (Recommended)
1. Open terminal/command prompt
2. Navigate to this folder: `cd path/to/mmf2mp4`
3. Run: `python install.py`
4. Follow any instructions shown

#### Option B: Manual Installation
1. Install Python packages: `pip install -r requirements.txt`
2. Install FFmpeg:
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

### Usage

#### Method 1: Double-click Scripts (Easiest)
- **Windows**: Double-click `convert.bat`
- **Mac/Linux**: Double-click `convert.sh`

These scripts will automatically start the interactive mode.

#### Method 2: Interactive Mode 
Just run the program without any arguments:
```bash
python mmf_decoder.py
```

The program will ask you:
- Path to your MMF file
- Where to save the MP4 file
- Frame rate (default: 20 FPS)
- Quality setting (default: 28, lower = better quality)
- Maximum frames (leave empty to convert all frames)

#### Method 3: Command Line Mode (Advanced)
If you're comfortable with command line:
```bash
python mmf_decoder.py input.mmf output.mp4 --fps 20 --quality 28
```

### Examples

**Convert a file with default settings:**
```bash
python mmf_decoder.py
# Then enter: /path/to/your/video.mmf
# Output will be: /path/to/your/video.mp4
```

**Convert with high quality:**
```bash
python mmf_decoder.py video.mmf video.mp4 --quality 18
```

**Create slow motion video:**
```bash
python mmf_decoder.py video.mmf video.mp4 --fps 10
```

**Test with first 100 frames only:**
```bash
python mmf_decoder.py video.mmf test.mp4 --max-frames 100
```

## ⚙️ Settings Explained

### Frame Rate (FPS)
- **10-15 FPS**: Slow motion effect
- **20-24 FPS**: Standard for scientific videos
- **30 FPS**: Smooth motion
- **Higher**: Smoother but larger file size

### Quality (CRF)
- **18-23**: High quality (larger files)
- **24-28**: Good quality (recommended)
- **29-35**: Lower quality (smaller files)
- **0**: Lossless (huge files)
- **51**: Worst quality (tiny files)

## 🔧 Troubleshooting

### "File not found" error
- Make sure the MMF file path is correct
- Use quotes around paths with spaces: `"C:\My Files\video.mmf"`
- Try dragging and dropping the file into the terminal

### "FFmpeg not found" error
- Install FFmpeg using the installation script: `python install.py`
- Or install manually following the links above

### Conversion is very slow
- Try converting just a few frames first: use `--max-frames 100`
- Use lower quality setting: `--quality 35`
- Check if your MMF file is very large

### Out of memory error
- The program loads data efficiently, but very large files might cause issues
- Try converting in smaller chunks using `--max-frames`

### Video looks wrong
- Check the frame rate: media data often needs lower FPS (10-20)
- Adjust quality settings if video is too pixelated or file too large

## 📁 File Formats

**Input**: MMF files (Media Memory File)
**Output**: MP4 files (H.264 codec, compatible with all video players)

## 💡 Tips for Lab Use

1. **Test first**: Always test with `--max-frames 100` before converting large files
2. **Batch processing**: You can create simple scripts to convert multiple files
3. **Backup**: Keep your original MMF files safe
4. **Quality**: Start with default quality (28), adjust if needed
5. **Frame rate**: 20 FPS works well for most media data

## 🆘 Getting Help

If you encounter issues:
1. Run `python install.py` to check your setup
2. Try the interactive mode first
3. Test with a small file using `--max-frames 100`
4. Check that your MMF file isn't corrupted

## 📋 System Requirements

- Python 3.7 or higher
- FFmpeg (for video encoding)
- Enough disk space (MP4 files are usually smaller than MMF)
- RAM: Depends on your MMF file size, but the program is memory-efficient

