#!/usr/bin/env python3
"""
Simple installation script for MMF Decoder
This script will install the required Python packages and check for FFmpeg
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("✗ Python 3.7 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True

def install_requirements():
    """Install Python requirements"""
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found")
        return False
    
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing Python packages")

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg is installed and available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ FFmpeg not found")
        print("\nFFmpeg is required for video conversion.")
        print("Please install FFmpeg:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  Mac: brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg (Ubuntu/Debian) or sudo yum install ffmpeg (CentOS/RHEL)")
        return False

def main():
    print("=" * 60)
    print("MMF Decoder Installation Script")
    print("=" * 60)
    print()
    
    success = True
    
    # Check Python version
    if not check_python():
        success = False
    
    print()
    
    # Install Python requirements
    if not install_requirements():
        success = False
    
    print()
    
    # Check FFmpeg
    if not check_ffmpeg():
        success = False
    
    print()
    print("=" * 60)
    
    if success:
        print("Installation completed successfully! 🎉")
        print("\nYou can now run the MMF decoder:")
        print("  python mmf_decoder.py")
        print("\nOr with command line arguments:")
        print("  python mmf_decoder.py input.mmf output.mp4")
    else:
        print("Installation completed with some issues.")
        print("Please resolve the issues above before using the decoder.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 