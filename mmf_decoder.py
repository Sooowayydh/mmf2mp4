#!/usr/bin/env python3
"""
MMF decoder
"""

import struct
import numpy as np
import subprocess
import argparse
import sys
from pathlib import Path

class AdaptiveMMFDecoder:
    
    def __init__(self, mmf_path):
        self.mmf_path = Path(mmf_path)
        self.file_size = self.mmf_path.stat().st_size
        
        # File structure
        self.header = None
        self.stacks = []
        self.total_frames = 0
        
        # Frame dimensions (determined from first stack)
        self.frame_width = None
        self.frame_height = None
        
        # Cache for loaded data
        self.stack_data = {}  # Cache for complete stack data
        
        print(f"Opening MMF file: {self.mmf_path.name}")
        print(f"File size: {self.file_size / (1024*1024):.1f} MB")
        
    def parse_file(self):
        """Parse the complete MMF file structure"""
        with open(self.mmf_path, 'rb') as f:
            # Read main header
            self.header = self._read_main_header(f)
            print(f"Header: {self.header['description'][:50]}...")
            
            # Parse all stacks
            frame_offset = 0
            while f.tell() < self.file_size:
                try:
                    stack_info = self._read_stack_header(f)
                    if stack_info is None:
                        break
                    
                    stack_info['start_frame'] = frame_offset
                    stack_info['end_frame'] = frame_offset + stack_info['nframes'] - 1
                    frame_offset += stack_info['nframes']
                    
                    # Get frame dimensions from first stack
                    if self.frame_width is None or self.frame_height is None:
                        current_pos = f.tell()
                        f.seek(stack_info['data_start'])
                        
                        # Read first image to get dimensions
                        test_image, file_info = self._read_ipl_image_with_widthstep(f)
                        if test_image is not None:
                            self.frame_height, self.frame_width = test_image.shape
                            print(f"Detected frame dimensions: {self.frame_width}x{self.frame_height}")
                        
                        f.seek(current_pos)
                    
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
                    print(f"Error parsing stack: {e}")
                    break
            
            self.total_frames = frame_offset
            print(f"Found {len(self.stacks)} stacks with {self.total_frames} total frames")
            if self.frame_width and self.frame_height:
                print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
    
    def _read_main_header(self, f):
        """Read the main MMF file header"""
        f.seek(0)
        pos = f.tell()
        
        # Read description until null
        description = b""
        while True:
            byte = f.read(1)
            if not byte or byte == b'\x00':
                break
            description += byte
        
        # Read header fields
        id_code = struct.unpack('<I', f.read(4))[0]
        header_size = struct.unpack('<I', f.read(4))[0]
        key_frame_interval = struct.unpack('<I', f.read(4))[0]
        thresh_above_bg = struct.unpack('<I', f.read(4))[0]
        thresh_below_bg = struct.unpack('<I', f.read(4))[0]
        
        # Seek to end of header
        f.seek(pos + header_size)
        
        return {
            'description': description.decode('utf-8', errors='ignore'),
            'id_code': id_code,
            'header_size': header_size,
            'key_frame_interval': key_frame_interval,
            'thresh_above_bg': thresh_above_bg,
            'thresh_below_bg': thresh_below_bg
        }
    
    def _read_stack_header(self, f):
        """Read an image stack header"""
        start_pos = f.tell()
        
        try:
            id_code = struct.unpack('<I', f.read(4))[0]
            header_size = struct.unpack('<I', f.read(4))[0]
            stack_size = struct.unpack('<I', f.read(4))[0]
            nframes = struct.unpack('<I', f.read(4))[0]
            
            if id_code != 0xbb67ca20:
                return None
                
            f.seek(start_pos + header_size)
            
            return {
                'id_code': id_code,
                'header_size': header_size,
                'stack_size': stack_size,
                'nframes': nframes,
                'file_position': start_pos,
                'data_start': start_pos + header_size
            }
            
        except:
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
        """Read IplImage with proper widthStep padding handling - ADAPTIVE DIMENSIONS"""
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
            
            print(f"    Image dimensions: {width}x{height}, depth: {depth}, widthStep: {widthStep}")
            
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
            bri_header = self._read_bri_header(f)
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
                    # Use actual frame dimensions instead of hardcoded 2048
                    if (x >= 0 and y >= 0 and 
                        x + width <= self.frame_width and 
                        y + height <= self.frame_height):
                        result[y:y+height, x:x+width] = subimage
            
            return result
            
        except Exception as e:
            print(f"    Error reading BRI frame: {e}")
            return background.copy()
    
    def _read_bri_header(self, f):
        """Read BRI header like Java BackgroundRemovedImageHeader"""
        try:
            pos = f.tell()
            
            # Read BRI magic number
            id_code = struct.unpack('<I', f.read(4))[0]
            if id_code != 0xf80921af:
                # Search for BRI magic in next bytes
                f.seek(pos)
                search_data = f.read(1000)
                bri_magic = struct.pack('<I', 0xf80921af)
                bri_pos = search_data.find(bri_magic)
                
                if bri_pos < 0:
                    return None
                    
                f.seek(pos + bri_pos)
                id_code = struct.unpack('<I', f.read(4))[0]
                
            header_size = struct.unpack('<I', f.read(4))[0]
            cv_depth = struct.unpack('<I', f.read(4))[0]
            channels = struct.unpack('<I', f.read(4))[0]
            num_subimages = struct.unpack('<I', f.read(4))[0]
            
            # Skip rest of header like Java
            bytes_read = 20  # We've read 5 ints
            remaining = header_size - bytes_read
            if remaining > 0:
                f.read(remaining)
            
            return {
                'id_code': id_code,
                'header_size': header_size,
                'cv_depth': cv_depth,
                'channels': channels,
                'num_subimages': num_subimages
            }
            
        except Exception as e:
            return None
    
    def _read_specific_bri_frame_cached(self, stack_idx, bri_frame_idx, background, file_info):
        """Read a specific BRI frame using cached positions (much faster)"""
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
                        self._skip_bri_frame(f)
                
                return background.copy()
                
        except Exception as e:
            print(f"    Error reading specific BRI frame: {e}")
            return background.copy()
    
    def _skip_bri_frame(self, f):
        """Skip a BRI frame without reading it"""
        try:
            bri_header = self._read_bri_header(f)
            if bri_header is None:
                return
            
            # Skip all sub-images
            for i in range(bri_header['num_subimages']):
                # Read rectangle dimensions
                x = struct.unpack('<I', f.read(4))[0]
                y = struct.unpack('<I', f.read(4))[0]
                width = struct.unpack('<I', f.read(4))[0]
                height = struct.unpack('<I', f.read(4))[0]
                
                # Skip pixel data
                bytes_per_pixel = 1  # Assume 8-bit for skipping
                pixel_data_size = width * height * bytes_per_pixel
                f.read(pixel_data_size)
                
        except Exception:
            pass
    
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
        """Fast BRI frame skipping with minimal reads"""
        try:
            pos = f.tell()
            
            # Try to find BRI magic quickly
            search_chunk = f.read(100)  # Small search chunk
            bri_magic = struct.pack('<I', 0xf80921af)
            bri_pos = search_chunk.find(bri_magic)
            
            if bri_pos < 0:
                # Search in larger chunk if not found
                f.seek(pos)
                search_chunk = f.read(1000)
                bri_pos = search_chunk.find(bri_magic)
                if bri_pos < 0:
                    return False
            
            # Jump to BRI header
            f.seek(pos + bri_pos + 4)  # Skip magic
            
            # Read header quickly
            header_size = struct.unpack('<I', f.read(4))[0]
            cv_depth = struct.unpack('<I', f.read(4))[0]
            channels = struct.unpack('<I', f.read(4))[0]
            num_subimages = struct.unpack('<I', f.read(4))[0]
            
            # Skip rest of header
            remaining_header = max(0, header_size - 20)
            f.read(remaining_header)
            
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
        """Convert MMF to MP4 with adaptive dimensions"""
        if self.frame_width is None or self.frame_height is None:
            print("Frame dimensions not detected")
            return False
            
        total_frames = min(self.total_frames, max_frames) if max_frames else self.total_frames
        
        print(f"Converting to MP4: {output_path}")
        print(f"Total frames: {total_frames}")
        print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
        print(f"Frame rate: {frame_rate} FPS")
        
        # FFmpeg command with adaptive dimensions
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{self.frame_width}x{self.frame_height}',
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
            
            # Stream frames
            for frame_num in range(total_frames):
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
                
                frame = self.read_frame(frame_num)
                if frame is not None:
                    process.stdin.write(frame.tobytes())
                else:
                    # Black frame fallback with correct dimensions
                    black_frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
                    process.stdin.write(black_frame.tobytes())
            
            process.stdin.close()
            process.wait()
            
            if process.returncode == 0:
                print(f"Conversion complete: {output_path}")
                return True
            else:
                print(f"FFmpeg failed with return code: {process.returncode}")
                return False
                
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Adaptive MMF to MP4 converter')
    parser.add_argument('input_mmf', help='Input MMF file')
    parser.add_argument('output_mp4', help='Output MP4 file')
    parser.add_argument('--fps', type=int, default=20, help='Frame rate (default: 20)')
    parser.add_argument('--quality', type=int, default=28, help='CRF quality 0-51 (default: 28)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to convert (for testing)')
    
    args = parser.parse_args()
    
    if not Path(args.input_mmf).exists():
        print(f"Input file not found: {args.input_mmf}")
        sys.exit(1)
    
    decoder = AdaptiveMMFDecoder(args.input_mmf)
    decoder.parse_file()
    
    success = decoder.convert_to_mp4(args.output_mp4, args.fps, args.quality, args.max_frames)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 