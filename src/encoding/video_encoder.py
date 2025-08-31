import subprocess
import os
import tempfile
from typing import Optional, Dict
from pathlib import Path

class VideoEncoder:
    """Handles video encoding with optimized settings"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="encoding_")
    
    def encode_final_video(self, input_video: str, output_video: str, 
                          target_bitrate: str = "8M") -> bool:
        """
        Encode final video with optimized settings for social media
        
        Args:
            input_video: Input video file
            output_video: Output video file  
            target_bitrate: Target bitrate (e.g., "8M", "6M")
            
        Returns:
            True if encoding successful
        """
        try:
            # Optimized FFmpeg command for social media
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', input_video,
                
                # Video encoding settings
                '-c:v', 'libx264',
                '-preset', 'slow',        # Better compression
                '-crf', '23',             # Good quality
                '-maxrate', target_bitrate,
                '-bufsize', '16M',        # Buffer size
                '-pix_fmt', 'yuv420p',    # Compatible format
                
                # Resolution and framerate
                '-vf', 'scale=1080:1920,fps=30',  # Ensure exact specs
                
                # Audio settings
                '-c:a', 'copy',
                
                # Optimization flags
                '-movflags', '+faststart',  # Fast streaming start
                '-avoid_negative_ts', 'make_zero',
                
                output_video
            ]
            
            print(f"Encoding video: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Video encoding successful")
                
                # Verify output
                if os.path.exists(output_video):
                    file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
                    print(f"✓ Output file: {output_video} ({file_size:.1f} MB)")
                    return True
                else:
                    print("✗ Output file not created")
                    return False
            else:
                print("✗ FFmpeg encoding failed")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Encoding error: {e}")
            return False
    
    def preprocess_input(self, input_video: str) -> str:
        """
        Preprocess input video to normalize format
        
        Returns:
            Path to preprocessed video
        """
        preprocessed_path = os.path.join(self.temp_dir, "preprocessed.mp4")
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video,
                '-vf', 'fps=30,scale=-2:1080',  # Normalize to 30fps, 1080p height
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '20',
                '-c:a', 'aac',
                '-b:a', '128k',
                preprocessed_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(preprocessed_path):
                print(f"✓ Input preprocessed: {preprocessed_path}")
                return preprocessed_path
            else:
                print("Preprocessing failed, using original")
                return input_video
                
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return input_video
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get detailed video information"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                
                # Extract video stream info
                video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
                
                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                        'duration': float(info['format'].get('duration', 0)),
                        'bitrate': int(info['format'].get('bit_rate', 0)),
                        'codec': video_stream.get('codec_name', 'unknown')
                    }
            
            return {}
            
        except Exception as e:
            print(f"Could not get video info: {e}")
            return {}
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass

# Test video encoding
def test_video_encoding():
    """Test video encoding pipeline"""

    encoder = VideoEncoder()
    
    input_video = "data/samples/s5.mp4"
    
    if not os.path.exists(input_video):
        print(f"Test video not found: {input_video}")
        return
    
    # Get video info
    info = encoder.get_video_info(input_video)
    print(f"Input video info: {info}")
    
    # Test preprocessing
    preprocessed = encoder.preprocess_input(input_video)
    print(f"Preprocessed: {preprocessed}")
    
    # Test encoding (create short test clip)
    test_output = "output/encoding_test.mp4"
    os.makedirs("output", exist_ok=True)
    
    # Create short clip for testing
    temp_clip = os.path.join(encoder.temp_dir, "test_clip.mp4")
    
    try:
        # Extract first 10 seconds for testing
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-t', '10',  # 10 seconds
            '-c', 'copy',
            temp_clip
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        if os.path.exists(temp_clip):
            success = encoder.encode_final_video(temp_clip, test_output, "6M")
            if success:
                print("✓ Video encoding test passed")
            else:
                print("✗ Video encoding test failed")
        
    except Exception as e:
        print(f"Encoding test error: {e}")
    
    encoder.cleanup()

if __name__ == "__main__":
    test_video_encoding()