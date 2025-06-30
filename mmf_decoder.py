#!/usr/bin/env python3
"""
MMF Decoder 
"""

import struct
import numpy as np
import subprocess
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

class FixedHeaderMMFDecoder:
    """MMF decoder with specification-compliant fixed header sizes"""
    
    def __init__(self, mmf_path):
        self.mmf_path = Path(mmf_path)
        self.file_size = self.mmf_path.stat().st_size
        
        # File structure
        self.header = None
        self.stacks = []
        self.total_frames = 0
        
        # Cache for loaded data
        self.stack_data = {}  # Cache for complete stack data
        
        print(f"Opening MMF file: {self.mmf_path.name}")
        print(f"File size: {self.file_size / (1024*1024):.1f} MB")
        
    def parse_file(self):
        """Parse the complete MMF file structure with fixed header sizes"""
        with open(self.mmf_path, 'rb') as f:
            # Read main header (FIXED: exactly 10240 bytes)
            self.header = self._read_main_header_fixed(f)
            print(f"Header: {self.header['description'][:50]}...")
            
            # Parse all stacks
            frame_offset = 0
            while f.tell() < self.file_size:
                try:
                    stack_info = self._read_stack_header_fixed(f)
                    if stack_info is None:
                        break
                    
                    stack_info['start_frame'] = frame_offset
                    stack_info['end_frame'] = frame_offset + stack_info['nframes'] - 1
                    frame_offset += stack_info['nframes']
                    
                    # Cache BRI frame positions for this stack (during parsing)
                    current_pos = f.tell()
                    f.seek(stack_info['data_start'])
                    bri_positions = self._cache_bri_positions(f, stack_info['nframes'])
                    stack_info['bri_positions'] = bri_positions
                    f.seek(current_pos)
                    
                    self.stacks.append(stack_info)
                    print(f"Stack {len(self.stacks)}: {stack_info['nframes']} frames (cached {len(bri_positions)} BRI positions)")
                    
                    # Skip to next stack
                    next_pos = stack_info['file_position'] + stack_info['stack_size']
                    if next_pos >= self.file_size:
                        break
                    f.seek(next_pos)
                    
                except Exception as e:
                    print(f"Warning: Error parsing stack: {e}")
                    break
            
            self.total_frames = frame_offset
            print(f"Found {len(self.stacks)} stacks with {self.total_frames} total frames")
    
    def _read_main_header_fixed(self, f):
        """Read main MMF header with FIXED 10240 byte size per specification"""
        f.seek(0)
        header_start = f.tell()
        
        # Read the entire 10240 byte header as per specification
        header_data = f.read(10240)
        if len(header_data) < 10240:
            raise ValueError("File too small for MMF main header (expected 10240 bytes)")
        
        # Find null terminator for description
        null_pos = header_data.find(b'\x00')
        if null_pos < 0:
            raise ValueError("Invalid MMF header: no null terminator found")
        
        description = header_data[:null_pos].decode('utf-8', errors='ignore')
        
        # Parse fixed structure after null terminator
        # Format: \0 + idcode + header_size + key_frame_interval + thresh_below + thresh_above
        data_start = null_pos + 1
        
        if data_start + 20 > len(header_data):  # Need at least 20 bytes for 5 ints
            raise ValueError("Invalid MMF header: insufficient data after description")
        
        # Unpack the fixed structure
        values = struct.unpack('<5I', header_data[data_start:data_start + 20])
        id_code, header_size, key_frame_interval, thresh_below_bg, thresh_above_bg = values
        
        # FIXED: Validate main header ID code per specification
        if id_code != 0xa3d2d45d:
            print(f"Warning: Unexpected main header ID: 0x{id_code:08x} (expected 0xa3d2d45d)")
        
        # FIXED: Validate header size matches specification
        if header_size != 10240:
            print(f"Warning: Header size mismatch: {header_size} (expected 10240)")
        
        # FIXED: Always position at exactly 10240 bytes regardless of claimed header size
        f.seek(header_start + 10240)
        
        return {
            'description': description,
            'id_code': id_code,
            'header_size': 10240,  # FIXED: Always use spec size
            'key_frame_interval': key_frame_interval,
            'thresh_below_bg': thresh_below_bg,
            'thresh_above_bg': thresh_above_bg
        }
    
    def _read_stack_header_fixed(self, f):
        """Read stack header with FIXED 512 byte size per specification"""
        start_pos = f.tell()
        
        try:
            # Read exactly 512 bytes per specification
            header_data = f.read(512)
            if len(header_data) < 512:
                return None
            
            # Parse fixed structure from header
            id_code, claimed_header_size, stack_size, nframes = struct.unpack('<4I', header_data[:16])
            
            # FIXED: Validate stack header ID code per specification
            if id_code != 0xbb67ca20:
                return None
            
            # FIXED: Validate header size
            if claimed_header_size != 512:
                print(f"Warning: Stack header size mismatch: {claimed_header_size} (expected 512)")
            
            return {
                'id_code': id_code,
                'header_size': 512,  # FIXED: Always use spec size
                'stack_size': stack_size,
                'nframes': nframes,
                'file_position': start_pos,
                'data_start': start_pos + 512  # FIXED: Always 512 bytes per spec
            }
            
        except Exception as e:
            print(f"Error reading stack header: {e}")
            return None
    
    def _read_bri_header_fixed(self, f):
        """Read BRI header with FIXED 1024 byte size per specification"""
        try:
            header_start = f.tell()
            
            # Read exactly 1024 bytes per specification
            header_data = f.read(1024)
            if len(header_data) < 1024:
                return None
            
            # Search for BRI magic number in header
            bri_magic = struct.pack('<I', 0xf80921af)
            magic_pos = header_data.find(bri_magic)
            
            if magic_pos < 0:
                return None
            
            # Parse from magic number position
            data_start = magic_pos
            if data_start + 20 > len(header_data):
                return None
            
            values = struct.unpack('<5I', header_data[data_start:data_start + 20])
            id_code, claimed_header_size, cv_depth, channels, num_subimages = values
            
            # FIXED: Validate BRI header ID code per specification
            if id_code != 0xf80921af:
                return None
            
            # FIXED: Validate header size
            if claimed_header_size != 1024:
                print(f"Warning: BRI header size mismatch: {claimed_header_size} (expected 1024)")
            
            # FIXED: Position is already at end of header (exactly 1024 bytes per spec)
            # No need to seek since we read exactly 1024 bytes
            
            return {
                'id_code': id_code,
                'header_size': 1024,  # FIXED: Always use spec size
                'cv_depth': cv_depth,
                'channels': channels,
                'num_subimages': num_subimages
            }
            
        except Exception as e:
            return None
    
    def _get_stack_background(self, stack_idx):
        """Load only the background for a stack (memory efficient)"""
        cache_key = f'bg_{stack_idx}'
        if cache_key in self.stack_data:
            bg_data = self.stack_data[cache_key]
            return bg_data['background'], bg_data['file_info']
            
        stack = self.stacks[stack_idx]
        print(f"Loading background for stack {stack_idx + 1}...")
        
        with open(self.mmf_path, 'rb') as f:
            f.seek(stack['data_start'])
            
            # Read only background image
            background, file_info = self._read_ipl_image_with_widthstep(f)
            if background is None:
                return None, None
            
            print(f"    Background: {background.shape}, range {np.min(background)}-{np.max(background)}")
            
            # Cache only background and file_info
            bg_data = {'background': background, 'file_info': file_info}
            self.stack_data[cache_key] = bg_data
            
            return background, file_info
    
    def _read_ipl_image_with_widthstep(self, f):
        """Read IplImage with proper widthStep padding handling"""
        try:
            start_pos = f.tell()
            
            # Read IplImage header fields exactly like Java
            nSize = struct.unpack('<I', f.read(4))[0]
            ID = struct.unpack('<I', f.read(4))[0]
            nChannels = struct.unpack('<I', f.read(4))[0]
            alphaChannel = struct.unpack('<I', f.read(4))[0]
            depth = struct.unpack('<I', f.read(4))[0]
            
            # Read colorModel and channelSeq as bytes (4 each)
            colorModel = [f.read(1)[0] for _ in range(4)]
            channelSeq = [f.read(1)[0] for _ in range(4)]
            
            dataOrder = struct.unpack('<I', f.read(4))[0]
            origin = struct.unpack('<I', f.read(4))[0]
            align = struct.unpack('<I', f.read(4))[0]
            width = struct.unpack('<I', f.read(4))[0]
            height = struct.unpack('<I', f.read(4))[0]
            
            # Read pointers and critical widthStep
            if nSize == 112:
                roiPTR = struct.unpack('<I', f.read(4))[0]
                maskROIPTR = struct.unpack('<I', f.read(4))[0]
                imageIdPTR = struct.unpack('<I', f.read(4))[0]
                tileInfoPTR = struct.unpack('<I', f.read(4))[0]
                imageSize = struct.unpack('<I', f.read(4))[0]
                imageDataPTR = struct.unpack('<I', f.read(4))[0]
                widthStep = struct.unpack('<I', f.read(4))[0]
            else:
                print(f"    Unsupported IplImage header size: {nSize}")
                return None, None
            
            # Skip rest of header
            bytes_read = f.tell() - start_pos
            remaining_header = nSize - bytes_read
            if remaining_header > 0:
                f.read(remaining_header)
            
            # Validate dimensions
            if width != 2048 or height != 2048:
                print(f"    Unexpected dimensions: {width}x{height}")
                return None, None
            
            # Determine bytes per pixel
            if depth == 8:
                bytes_per_pixel = 1
                dtype = np.uint8
            elif depth == 16:
                bytes_per_pixel = 2
                dtype = np.uint16
            else:
                print(f"    Unsupported depth: {depth}")
                return None, None
            
            # Create file_info for BRI reading
            file_info = {
                'width': width,
                'height': height,
                'depth': depth,
                'bytes_per_pixel': bytes_per_pixel,
                'widthStep': widthStep
            }
            
            # **CRITICAL**: Handle widthStep correctly like Java code
            expected_row_size = width * bytes_per_pixel
            
            if expected_row_size == widthStep:
                # No padding - read contiguously
                pixel_data = f.read(width * height * bytes_per_pixel)
                pixels = np.frombuffer(pixel_data, dtype=dtype)
            else:
                # Padding present - read row by row like Java
                all_pixels = []
                for row in range(height):
                    # Read full row including padding
                    row_data = f.read(widthStep)
                    if len(row_data) < widthStep:
                        print(f"    Insufficient data for row {row}")
                        return None, None
                    
                    # Extract only the actual pixel data
                    pixel_row = row_data[:expected_row_size]
                    row_pixels = np.frombuffer(pixel_row, dtype=dtype)
                    all_pixels.extend(row_pixels)
                
                pixels = np.array(all_pixels, dtype=dtype)
            
            # Reshape to image
            image = pixels.reshape((height, width))
            
            # Convert to 8-bit if needed
            if dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            
            return image, file_info
                        
        except Exception as e:
            print(f"    Error reading IplImage: {e}")
            return None, None
    
    def _read_bri_frame_complete(self, f, background, file_info):
        """Read complete BRI frame exactly like Java BackgroundRemovedImage"""
        try:
            # Read BRI header like Java BackgroundRemovedImageHeader
            bri_header = self._read_bri_header_fixed(f)
            if bri_header is None:
                return background.copy()
            
            # Start with background copy
            result = background.copy()
            
            # Read and apply each sub-image like Java
            for i in range(bri_header['num_subimages']):
                subimage_data = self._read_bri_subimage(f, file_info)
                if subimage_data is not None:
                    x, y, width, height, subimage = subimage_data
                    
                    # Apply sub-image to result (like Java insertIntoImage)
                    if (x >= 0 and y >= 0 and 
                        x + width <= 2048 and 
                        y + height <= 2048):
                        result[y:y+height, x:x+width] = subimage
            
            return result
            
        except Exception as e:
            print(f"    Error reading BRI frame: {e}")
            return background.copy()
    
    def _read_bri_subimage(self, f, file_info):
        """Read BRI sub-image exactly like Java readBRISubIm"""
        try:
            # Read rectangle like Java
            x = struct.unpack('<I', f.read(4))[0]
            y = struct.unpack('<I', f.read(4))[0]
            width = struct.unpack('<I', f.read(4))[0]
            height = struct.unpack('<I', f.read(4))[0]
            
            # Read pixel data like Java
            bytes_per_pixel = file_info['bytes_per_pixel']
            pixel_data_size = width * height * bytes_per_pixel
            
            pixel_data = f.read(pixel_data_size)
            if len(pixel_data) < pixel_data_size:
                return None
            
            # Parse pixels based on depth like Java
            if file_info['depth'] == 8:
                pixels = np.frombuffer(pixel_data, dtype=np.uint8)
            elif file_info['depth'] == 16:
                pixels = np.frombuffer(pixel_data, dtype=np.uint16)
                pixels = (pixels / 256).astype(np.uint8)  # Convert to 8-bit
            else:
                return None
            
            # Reshape to sub-image
            subimage = pixels.reshape((height, width))
            
            return x, y, width, height, subimage
            
        except Exception as e:
            return None
    
    def _cache_bri_positions(self, f, frames_in_stack):
        """Cache the file positions of all BRI frames in this stack for fast seeking"""
        positions = []
        
        try:
            # Skip background image
            bg_skip, _ = self._read_ipl_image_with_widthstep(f)
            if bg_skip is None:
                return positions
            
            # Cache position of each BRI frame
            for frame_idx in range(frames_in_stack - 1):  # -1 because first frame is background
                pos = f.tell()
                positions.append(pos)
                
                # Skip this BRI frame quickly
                if not self._skip_bri_frame_fast(f):
                    break
                    
        except Exception as e:
            print(f"    Warning: Could not cache all BRI positions: {e}")
            
        return positions

    def _skip_bri_frame_fast(self, f):
        """Fast BRI frame skipping with FIXED header sizes"""
        try:
            # FIXED: Skip exactly 1024 bytes for BRI header per spec
            header_start = f.tell()
            header_data = f.read(1024)
            
            if len(header_data) < 1024:
                return False
            
            # Find and parse BRI header quickly
            bri_magic = struct.pack('<I', 0xf80921af)
            magic_pos = header_data.find(bri_magic)
            
            if magic_pos < 0:
                return False
            
            # Parse header quickly
            data_start = magic_pos
            if data_start + 20 > len(header_data):
                return False
            
            values = struct.unpack('<5I', header_data[data_start:data_start + 20])
            id_code, header_size, cv_depth, channels, num_subimages = values
            
            if id_code != 0xf80921af:
                return False
            
            # Skip all sub-images quickly
            bytes_per_pixel = 2 if cv_depth == 16 else 1
            
            for i in range(num_subimages):
                # Read rectangle dims
                x = struct.unpack('<I', f.read(4))[0]
                y = struct.unpack('<I', f.read(4))[0]
                width = struct.unpack('<I', f.read(4))[0]
                height = struct.unpack('<I', f.read(4))[0]
                
                # Skip pixel data
                pixel_data_size = width * height * bytes_per_pixel
                f.read(pixel_data_size)
            
            return True
            
        except Exception:
            return False

    def _read_specific_bri_frame_cached(self, stack_idx, bri_frame_idx, background, file_info):
        """Read a specific BRI frame using cached positions (much faster!)"""
        try:
            stack = self.stacks[stack_idx]
            
            # Check if we have cached position
            if (bri_frame_idx < len(stack['bri_positions'])):
                position = stack['bri_positions'][bri_frame_idx]
                
                with open(self.mmf_path, 'rb') as f:
                    f.seek(position)
                    return self._read_bri_frame_complete(f, background, file_info)
            else:
                # Fallback to sequential reading
                return self._read_specific_bri_frame(stack_idx, bri_frame_idx, background, file_info)
                
        except Exception as e:
            print(f"    Error reading cached BRI frame: {e}")
            return background.copy()
    
    def _read_specific_bri_frame(self, stack_idx, bri_frame_idx, background, file_info):
        """Read a specific BRI frame without loading all frames"""
        try:
            stack = self.stacks[stack_idx]
            
            with open(self.mmf_path, 'rb') as f:
                f.seek(stack['data_start'])
                
                # Skip background
                bg_skip, _ = self._read_ipl_image_with_widthstep(f)
                if bg_skip is None:
                    return background.copy()
                
                # Skip to target BRI frame
                for i in range(bri_frame_idx + 1):
                    if i == bri_frame_idx:
                        # Read this frame
                        return self._read_bri_frame_complete(f, background, file_info)
                    else:
                        # Skip this frame by reading header and jumping
                        self._skip_bri_frame_fast(f)
                
                return background.copy()
                
        except Exception as e:
            print(f"    Error reading specific BRI frame: {e}")
            return background.copy()
    
    def read_frame(self, frame_num):
        """Read and reconstruct a specific frame (memory efficient with caching)"""
        if frame_num < 0 or frame_num >= self.total_frames:
            return None
            
        # Find target stack
        target_stack_idx = None
        for i, stack in enumerate(self.stacks):
            if stack['start_frame'] <= frame_num <= stack['end_frame']:
                target_stack_idx = i
                break
                
        if target_stack_idx is None:
            return None
            
        # Get background for this stack (cached)
        background, file_info = self._get_stack_background(target_stack_idx)
        if background is None:
            return None
            
        target_stack = self.stacks[target_stack_idx]
        frame_in_stack = frame_num - target_stack['start_frame']
        
        if frame_in_stack == 0:
            # First frame is always the background
            return background.copy()
        else:
            # Read specific BRI frame using cached positions (much faster!)
            return self._read_specific_bri_frame_cached(target_stack_idx, frame_in_stack - 1, background, file_info)
    
    def convert_to_mp4(self, output_path, frame_rate=30, quality=28, max_frames=None):
        """Convert MMF to MP4"""
        total_frames = min(self.total_frames, max_frames) if max_frames else self.total_frames
        
        print(f"Converting to MP4: {output_path}")
        print(f"Total frames: {total_frames}")
        print(f"Frame rate: {frame_rate} FPS")
        
        # FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', '2048x2048',
            '-r', str(frame_rate),
            '-i', '-',
            '-c:v', 'libx264',
            '-crf', str(quality),
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            str(output_path)
        ]
        
        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            
            # Stream frames with progress bar
            with tqdm(total=total_frames, desc="Converting frames", unit="frame") as pbar:
                for frame_num in range(total_frames):
                    frame = self.read_frame(frame_num)
                    if frame is not None:
                        process.stdin.write(frame.tobytes())
                    else:
                        # Black frame fallback
                        black_frame = np.zeros((2048, 2048), dtype=np.uint8)
                        process.stdin.write(black_frame.tobytes())
                    
                    pbar.update(1)
            
            process.stdin.close()
            process.wait()
            
            if process.returncode == 0:
                print(f"Conversion complete: {output_path}")
                return True
            else:
                print(f"Error: FFmpeg failed with return code: {process.returncode}")
                return False
                
        except Exception as e:
            print(f"Error: Conversion failed: {e}")
            return False

def interactive_mode():
    """Interactive mode for users who prefer prompts over command line arguments"""
    print("=" * 50)
    print("MMF to MP4 Converter - Interactive Mode")
    print("=" * 50)
    print()
    
    # Get input file
    while True:
        input_file = input("Enter the path to your MMF file: ").strip().strip('"\'')
        if not input_file:
            print("Please enter a file path.")
            continue
            
        input_path = Path(input_file)
        if input_path.exists():
            break
        else:
            print(f"File not found: {input_file}")
            print("Please check the path and try again.")
            print()
    
    # Get output file
    print()
    default_output = str(input_path.with_suffix('.mp4'))
    output_file = input(f"Enter output MP4 file path (or press Enter for '{default_output}'): ").strip().strip('"\'')
    if not output_file:
        output_file = default_output
    
    # Get settings with explanations
    print()
    print("Optional Settings (press Enter to use defaults):")
    print()
    
    # Frame rate with validation
    while True:
        fps_input = input("Frame rate (FPS) [default: 20]: ").strip()
        if not fps_input:
            fps = 20
            break
        try:
            fps = int(fps_input)
            if fps <= 0:
                print("Frame rate must be a positive number. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for frame rate.")
    
    # Quality with validation
    while True:
        quality_input = input("Quality (0-51, lower=better quality) [default: 28]: ").strip()
        if not quality_input:
            quality = 28
            break
        try:
            quality = int(quality_input)
            if quality < 0 or quality > 51:
                print("Quality must be between 0 and 51. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for quality (0-51).")
    
    # Max frames with validation
    while True:
        max_frames_input = input("Maximum frames to convert (for testing, leave empty for all): ").strip()
        if not max_frames_input:
            max_frames = None
            break
        try:
            max_frames = int(max_frames_input)
            if max_frames <= 0:
                print("Maximum frames must be a positive number. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for maximum frames.")
    
    # Confirm settings
    print()
    print("=" * 50)
    print("Conversion Settings:")
    print(f"Input file:     {input_file}")
    print(f"Output file:    {output_file}")
    print(f"Frame rate:     {fps} FPS")
    print(f"Quality:        {quality} (0-51 scale)")
    if max_frames:
        print(f"Max frames:     {max_frames}")
    else:
        print(f"Max frames:     All frames")
    print("=" * 50)
    print()
    
    confirm = input("Start conversion? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Conversion cancelled.")
        return
    
    print()
    print("Starting conversion...")
    print()
    
    try:
        decoder = FixedHeaderMMFDecoder(input_file)
        decoder.parse_file()
        
        success = decoder.convert_to_mp4(output_file, fps, quality, max_frames)
        
        if success:
            print()
            print("=" * 50)
            print("Conversion completed successfully!")
            print(f"Output file: {output_file}")
            print("=" * 50)
        else:
            print()
            print("Conversion failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Conversion failed.")

def main():
    # Check if no arguments provided - use interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return
    
    parser = argparse.ArgumentParser(description='MMF to MP4 converter with fixed header sizes')
    parser.add_argument('input_mmf', help='Input MMF file')
    parser.add_argument('output_mp4', help='Output MP4 file')
    parser.add_argument('--fps', type=int, default=20, help='Frame rate (default: 20)')
    parser.add_argument('--quality', type=int, default=28, help='CRF quality 0-51 (default: 28)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to convert (for testing)')
    
    args = parser.parse_args()
    
    if not Path(args.input_mmf).exists():
        print(f"Error: Input file not found: {args.input_mmf}")
        sys.exit(1)
    
    decoder = FixedHeaderMMFDecoder(args.input_mmf)
    decoder.parse_file()
    
    success = decoder.convert_to_mp4(args.output_mp4, args.fps, args.quality, args.max_frames)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 