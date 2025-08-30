import cv2
import os
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Callable

from .detectors.face_mp import FaceDetector
from .detectors.zone_detector import UniversalWebinarDetector
from .detectors.shot_detector import ShotChangeDetector, ConflictResolver
from .trackers.temporal import AdvancedEMATracker, TemporalMedianTracker, StabilityAnalyzer
from .layout.composer import EnhancedLayoutComposer
from .encoding.video_encoder import VideoEncoder
from .optimization.performance import PerformanceOptimizer

class ProductionPipeline:
   """Production-ready webinar reframing pipeline"""
   
   def __init__(self, config: Dict = None):
       self.config = config or self._default_config()
       self.performance_optimizer = PerformanceOptimizer()
       
       # Initialize components with optimizations
       self.performance_optimizer.optimize_opencv_settings()
       
       self.face_detector = FaceDetector(min_confidence=self.config['min_face_conf'])
       self.region_detector = UniversalWebinarDetector()
       self.shot_detector = ShotChangeDetector(threshold=self.config['shot_threshold'])
       self.conflict_resolver = ConflictResolver()
       
       # Enhanced trackers
       self.face_tracker = AdvancedEMATracker(
           alpha=self.config['face_alpha'], 
           adaptive=True
       )
       self.slide_tracker = TemporalMedianTracker(
           window_size=self.config['slide_window']
       )
       self.stability_analyzer = StabilityAnalyzer()
       
       # Enhanced composer and encoder
       self.layout_composer = EnhancedLayoutComposer()
       self.video_encoder = VideoEncoder()
   
   def process_video_production(self, input_path: str, output_path: str,
                              progress_callback: Optional[Callable] = None) -> bool:
       """
       Production video processing with quality optimizations
       """
       try:
           print(f"Starting production processing: {input_path}")
           
           # Step 1: Validate and preprocess input
           if not self._validate_input(input_path):
               return False
           
           # Get video information
           video_info = self.video_encoder.get_video_info(input_path)
           print(f"Input: {video_info.get('width', 0)}x{video_info.get('height', 0)} @ {video_info.get('fps', 0)} fps")
           
           # Step 2: Preprocess if needed
           processed_input = input_path
           if self.config.get('preprocess_input', True):
               processed_input = self.video_encoder.preprocess_input(input_path)
           
           # Step 3: Process frames with enhanced composition
           temp_output = self._process_frames_enhanced(processed_input, progress_callback)
           
           if not temp_output:
               return False
           
           # Step 4: Final encoding optimization
           success = self.video_encoder.encode_final_video(
               temp_output, output_path, self.config['target_bitrate']
           )

           # Debug: Check if final video has audio
           if success:
               try:
                   probe_cmd = ['ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'a', output_path]
                   probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                   if 'codec_type=audio' in probe_result.stdout:
                       print("Final video has audio")
                   else:
                       print("Final video missing audio!")
               except:
                   pass
           
           if success:
               self._log_final_stats(output_path)
           
           return success
           
       except Exception as e:
           print(f"Production pipeline error: {e}")
           return False
       finally:
           self._cleanup()
   
   def _validate_input(self, input_path: str) -> bool:
       """Validate input video"""
       if not os.path.exists(input_path):
           print(f"Input file not found: {input_path}")
           return False
       
       # Check file size
       file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
       if file_size_mb > 2000:  # > 2GB
           print(f"Warning: Large input file ({file_size_mb:.1f} MB)")
       
       return True
   
   def _process_frames_enhanced(self, input_path: str, 
                              progress_callback: Optional[Callable] = None) -> Optional[str]:
       """Enhanced frame processing with quality optimizations"""
       
       cap = cv2.VideoCapture(input_path)
       if not cap.isOpened():
           return None
       
       # Video properties
       fps = cap.get(cv2.CAP_PROP_FPS)
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       # Create temporary output
       temp_output = os.path.join(self.video_encoder.temp_dir, "composed.mp4")
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out_writer = cv2.VideoWriter(temp_output, fourcc, fps, (1080, 1920))
       
       print(f"Processing {total_frames} frames...")
       
       frame_count = 0
       start_time = time.time()
       last_shot_change = -10
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           frame_count += 1
           frame_start_time = time.time()
           
           # Detect shot changes
           is_shot_change = self.shot_detector.detect_shot_change(frame, frame_count)
           if is_shot_change:
               # Reset trackers on shot change
               self.face_tracker = AdvancedEMATracker(alpha=self.config['face_alpha'], adaptive=True)
               self.slide_tracker = TemporalMedianTracker(window_size=self.config['slide_window'])
               last_shot_change = frame_count
               print(f"Shot change detected at frame {frame_count}")
           
           # Detect regions
           face_bbox, face_conf = self.face_detector.detect(frame)
           face_region, slide_region = self.region_detector.get_regions(frame, face_bbox)
           
           # Apply enhanced tracking
           stable_face = self.face_tracker.update(face_region, face_conf)
           stable_slide = self.slide_tracker.update(slide_region)
           
           # Resolve conflicts
           if stable_face and stable_slide:
               stable_face, stable_slide = self.conflict_resolver.resolve_conflicts(
                   stable_face, stable_slide, frame.shape[:2]
               )
           
           # Analyze stability
           stability_metrics = self.stability_analyzer.analyze_stability(stable_face)
           
           # Compose frame with quality enhancements
           enhance_quality = frame_count > last_shot_change + 10  # Skip enhancement near cuts
           composed_frame = self.layout_composer.compose_frame(
               frame, stable_face, stable_slide, enhance_quality
           )
           
           # Write frame
           out_writer.write(composed_frame)
           
           # Performance monitoring
           frame_end_time = time.time()
           self.performance_optimizer.frame_times.append(frame_end_time - frame_start_time)
           self.performance_optimizer.monitor_performance(start_time, frame_count)
           
           # Progress reporting
           if frame_count % 100 == 0:
               self.performance_optimizer.log_performance(frame_count)
               
               if progress_callback:
                   progress = (frame_count / total_frames) * 100
                   progress_callback(progress, frame_count, total_frames)
               else:
                   progress = (frame_count / total_frames) * 100
                   print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
       
       cap.release()
       out_writer.release()
       
       # Add audio from original video
       print("Adding audio from original video...")
       try:
           temp_with_audio = os.path.join(self.video_encoder.temp_dir, "composed_with_audio.mp4")
           
           cmd = [
               'ffmpeg', '-y',
               '-i', temp_output,      # Video only
               '-i', input_path,       # Original with audio
               '-c:v', 'copy',
               '-c:a', 'copy',
               '-shortest',
               temp_with_audio
           ]
           
           result = subprocess.run(cmd, capture_output=True, text=True)
           
           if result.returncode == 0:
               print("Audio successfully merged")
               # Clean up video-only file
               try:
                   os.remove(temp_output)
               except:
                   pass
               return temp_with_audio
           else:
               print("Audio merge failed, proceeding without audio")
               print(f"FFmpeg error: {result.stderr}")
               return temp_output
               
       except Exception as e:
           print(f"Audio merge error: {e}")
           return temp_output
       
       print(f"Frame processing complete: {temp_output}")
       return temp_output
   
   def _log_final_stats(self, output_path: str):
       """Log final processing statistics"""
       try:
           file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
           
           print(f"\n=== Production Processing Complete ===")
           print(f"Output file: {output_path}")
           print(f"File size: {file_size_mb:.1f} MB")
           print(f"Processing FPS: {self.performance_optimizer.metrics.fps_processing:.1f}")
           print(f"Memory used: {self.performance_optimizer.metrics.memory_usage_mb:.1f} MB")
           
           # Performance suggestions
           suggestions = self.performance_optimizer.suggest_optimizations()
           if suggestions:
               print("\nOptimization suggestions for next run:")
               for suggestion in suggestions:
                   print(f"  • {suggestion}")
       
       except Exception as e:
           print(f"Could not log final stats: {e}")
   
   def _default_config(self) -> Dict:
       """Production-optimized default configuration"""
       return {
           'min_face_conf': 0.3,
           'shot_threshold': 0.3,
           'face_alpha': 0.3,
           'slide_window': 8,
           'target_bitrate': '8M',
           'preprocess_input': True,
           'enable_quality_enhancement': True,
           'enable_stability_analysis': True
       }
   
   def _cleanup(self):
       """Clean up resources"""
       try:
           self.face_detector.close()
           self.video_encoder.cleanup()
       except:
           pass

# Test production pipeline
def test_production_pipeline():
   """Test production-ready pipeline"""
   
   config = {
       'min_face_conf': 0.3,
       'shot_threshold': 0.3,
       'face_alpha': 0.3,
       'slide_window': 8,
       'target_bitrate': '6M'
   }
   
   pipeline = ProductionPipeline(config)
   
   def progress_callback(percent, current, total):
       print(f"Production Progress: {percent:.1f}% ({current}/{total})")
   
   input_video = "data/samples/s7.mp4"
   output_video = "output/production_test.mp4"
   
   os.makedirs("output", exist_ok=True)
   
   success = pipeline.process_video_production(
       input_video, output_video, progress_callback
   )
   
   return success

if __name__ == "__main__":
   test_production_pipeline()