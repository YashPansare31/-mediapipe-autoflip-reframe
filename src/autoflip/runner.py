import subprocess
import json
import os
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

class AutoFlipRunner:
    """Handles running AutoFlip with external hints"""
    
    def __init__(self, autoflip_binary: str = "autoflip"):
        """
        Args:
            autoflip_binary: Path to AutoFlip binary or command
        """
        self.autoflip_binary = autoflip_binary
        self.temp_dir = tempfile.mkdtemp(prefix="autoflip_")
    
    def run_autoflip(self, input_video: str, output_video: str, 
                     config_path: str, hints_path: Optional[str] = None) -> bool:
        """
        Run AutoFlip with configuration and optional external hints
        
        Args:
            input_video: Input video file path
            output_video: Output video file path  
            config_path: AutoFlip config file path
            hints_path: Optional external hints JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build AutoFlip command
            cmd = [
                self.autoflip_binary,
                f"--input_video={input_video}",
                f"--output_video={output_video}", 
                f"--config={config_path}"
            ]
            
            if hints_path and os.path.exists(hints_path):
                cmd.append(f"--external_hints={hints_path}")
                print(f"Using external hints: {hints_path}")
            
            print(f"Running AutoFlip: {' '.join(cmd)}")
            
            # Run AutoFlip
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("AutoFlip completed successfully")
                return True
            else:
                print(f"AutoFlip failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("AutoFlip timed out after 5 minutes")
            return False
        except FileNotFoundError:
            print(f"AutoFlip binary not found: {self.autoflip_binary}")
            print("Note: AutoFlip setup required for full pipeline")
            return False
        except Exception as e:
            print(f"Error running AutoFlip: {e}")
            return False
    
    def create_simple_config(self, output_path: str) -> str:
        """
        Create a simple AutoFlip config for 9:16 output
        
        Returns:
            Path to created config file
        """
        config_content = f"""
# Simple 9:16 AutoFlip configuration
target_aspect_ratio: 0.5625  # 9:16 = 1080/1920

# Output dimensions
output_width: 1080
output_height: 1920

# Enable external signals
use_external_hints: true

# Basic cropping parameters
crop_parameters {{
  target_size_type: MAXIMIZE_TARGET_DIMENSION
  crop_window_width_fraction: 1.0
  crop_window_height_fraction: 1.0
  overlap_allowance: 0.2
}}

# Motion options
motion_options {{
  tracking_options {{
    max_num_motions: 3
    min_motion_to_reframe: 0.3
  }}
}}
"""
        
        config_path = os.path.join(self.temp_dir, "simple_916.config")
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Created simple config: {config_path}")
        return config_path
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass

# Fallback: Manual composition when AutoFlip not available
class ManualComposer:
    """Manual frame composition as AutoFlip fallback"""
    
    def compose_video(self, input_video: str, output_video: str,
                     face_detector, region_detector) -> bool:
        """
        Manually compose video frames when AutoFlip unavailable
        """
        import cv2
        from ..layout.composer import LayoutComposer
        from ..trackers.temporal import EMATracker, TemporalMedianTracker
        
        try:
            # Initialize components
            composer = LayoutComposer()
            face_tracker = EMATracker(alpha=0.3)
            slide_tracker = TemporalMedianTracker(window_size=8)
            
            # Open input video
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (1080, 1920))
            
            print(f"Manual composition: {total_frames} frames at {fps} fps")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect regions
                face_bbox, face_conf = face_detector.detect(frame)
                face_region, slide_region = region_detector.get_regions(frame, face_bbox)
                
                # Apply temporal smoothing
                stable_face = face_tracker.update(face_region, face_conf)
                stable_slide = slide_tracker.update(slide_region)
                
                # Compose frame
                composed = composer.compose_frame(frame, stable_face, stable_slide)
                
                # Write frame
                out.write(composed)
                
                # Progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
            
            cap.release()
            out.release()
            
            print(f"Manual composition complete: {output_video}")
            return True
            
        except Exception as e:
            print(f"Manual composition failed: {e}")
            return False

# Test AutoFlip integration
def test_autoflip_integration():
    """Test AutoFlip runner and fallback"""
    runner = AutoFlipRunner()
    
    # Test config creation
    config_path = runner.create_simple_config("test_config.config")
    print(f"Config created: {os.path.exists(config_path)}")
    
    # Test AutoFlip execution (will likely fail without proper setup)
    success = runner.run_autoflip(
        input_video="data/samples/s5.mp4",
        output_video="test_autoflip_out.mp4", 
        config_path=config_path
    )
    
    if not success:
        print("AutoFlip not available, will use manual composition fallback")
    
    runner.cleanup()

if __name__ == "__main__":
    test_autoflip_integration()