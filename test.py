#!/usr/bin/env python3
"""Diagnose why OpenCV can't open the video"""

import cv2
import subprocess
import sys
from pathlib import Path

def diagnose_video(video_path):
    video_path = Path(video_path)
    
    print("=== Basic File Checks ===")
    print(f"Path: {video_path}")
    print(f"Exists: {video_path.exists()}")
    print(f"Is file: {video_path.is_file()}")
    print(f"Readable: {video_path.stat().st_mode if video_path.exists() else 'N/A'}")
    
    if video_path.exists():
        print(f"Size: {video_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Check file permissions
        import os
        print(f"Access (R_OK): {os.access(video_path, os.R_OK)}")
    
    print("\n=== OpenCV Build Info ===")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Video backend: {cv2.getBuildInformation()}")
    
    # Try different backends
    print("\n=== Testing Different Backends ===")
    backends = [
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY, "ANY"),
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTrying {backend_name}...")
        vcap = cv2.VideoCapture(str(video_path), backend_id)
        opened = vcap.isOpened()
        print(f"  Opened: {opened}")
        if opened:
            print(f"  Frames: {int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            print(f"  Size: {int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"  FPS: {vcap.get(cv2.CAP_PROP_FPS)}")
        vcap.release()
    
    # Use ffprobe to check if file is valid
    print("\n=== FFprobe Check ===")
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 
             'stream=codec_name,width,height,r_frame_rate,nb_frames',
             '-of', 'default=noprint_wrappers=1', str(video_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ FFprobe can read the file:")
            print(result.stdout)
        else:
            print("✗ FFprobe failed:")
            print(result.stderr)
    except FileNotFoundError:
        print("✗ ffprobe not found (install ffmpeg)")
    except Exception as e:
        print(f"✗ Error running ffprobe: {e}")
    
    # Check file header
    print("\n=== File Header Check ===")
    if video_path.exists():
        with open(video_path, 'rb') as f:
            header = f.read(12)
            print(f"First 12 bytes: {header.hex()}")
            print(f"ASCII interpretation: {header}")
            
            # Common video file signatures
            if header[4:8] == b'ftyp':
                print("✓ Looks like a valid MP4 file (ftyp signature found)")
            else:
                print("✗ Does not have standard MP4 signature")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_video.py <video_path>")
        sys.exit(1)
    
    diagnose_video(sys.argv[1])