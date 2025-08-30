import cv2
import os
from typing import Optional, Tuple
from pathlib import Path

from .detectors.face_mp import FaceDetector
from .detectors.zone_detector import UniversalWebinarDetector
from .detectors.shot_detector import ShotChangeDetector, ConflictResolver
from .trackers.temporal import EMATracker, TemporalMedianTracker
from .layout.composer import LayoutComposer
from .autoflip.hints import AutoFlipHints
from .autoflip.runner import AutoFlipRunner, ManualComposer

class WebinarReframingPipeline:
    """Complete end-to-end webinar reframing pipeline"""
    
    def __init__(self, config: dict = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Initialize components
        self.face_detector = FaceDetector(
            min_confidence=self.config.get('min_face_conf', 0.3)
        )
        self.region_detector = UniversalWebinarDetector()
        self.shot_detector = ShotChangeDetector(
            threshold=self.config.get('shot_threshold', 0.3)
        )
        self.conflict_resolver = ConflictResolver()
        
        # Trackers
        self.face_tracker = EMATracker(alpha=self.config.get('face_alpha', 0.3))
        self.slide_tracker = TemporalMedianTracker(
            window_size=self.config.get('slide_window', 8)
        )
        
        # Composers
        self.layout_composer = LayoutComposer()
        self.autoflip_hints = AutoFlipHints()
        self.autoflip_runner = AutoFlipRunner()
    
    def process_video(self, input_path: str, output_path: str, 
                     use_autoflip: bool = True) -> bool:
        """
        Process complete video through reframing pipeline
        
        Args:
            input_path: Input video file path
            output_path: Output video file path
            use_autoflip: Whether to use AutoFlip or manual composition
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Starting video processing: {input_path} -> {output_path}")
            
            # Validate input
            if not os.path.exists(input_path):
                print(f"Input file not found: {input_path}")
                return False
            
            # Step 1: Extract frame-level detections and hints
            if use_autoflip:
                hints_path = self._extract_hints(input_path)
                if not hints_path:
                    print("Hint extraction failed, falling back to manual composition")
                    use_autoflip = False
            
            # Step 2: Apply reframing
            if use_autoflip:
                success = self._process_with_autoflip(input_path, output_path, hints_path)
                if not success:
                    print("AutoFlip failed, trying manual composition")
                    use_autoflip = False
            
            if not use_autoflip:
                success = self._process_with_manual_composition(input_path, output_path)
            
            if success:
                print(f"Video processing complete: {output_path}")
                return True
            else:
                print("Video processing failed")
                return False
                
        except Exception as e:
            print(f"Pipeline error: {e}")
            return False
        finally:
            self._cleanup()
    
    def _extract_hints(self, input_path: str) -> Optional[str]:
        """Extract detection hints for AutoFlip"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Extracting hints from {total_frames} frames...")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Detect regions
            face_bbox, face_conf = self.face_detector.detect(frame)
            face_region, slide_region = self.region_detector.get_regions(frame, face_bbox)
            
            # Apply tracking for stability
            stable_face = self.face_tracker.update(face_region, face_conf)
            stable_slide = self.slide_tracker.update(slide_region)
            
            # Resolve conflicts
            if stable_face and stable_slide:
                stable_face, stable_slide = self.conflict_resolver.resolve_conflicts(
                    stable_face, stable_slide, frame.shape[:2]
                )
            
            # Add to hints
            self.autoflip_hints.add_frame_detections(
                frame_count, timestamp, stable_face, face_conf,
                stable_slide, width, height
            )
            
            # Progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Hint extraction: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        
        # Save hints
        hints_path = self.autoflip_hints.save_hints("extraction_hints.json")
        return hints_path
    
    def _process_with_autoflip(self, input_path: str, output_path: str, 
                              hints_path: str) -> bool:
        """Process video using AutoFlip"""
        # Create or use existing config
        config_path = self.config.get('autoflip_config')
        if not config_path:
            config_path = self.autoflip_runner.create_simple_config()
        
        return self.autoflip_runner.run_autoflip(
            input_path, output_path, config_path, hints_path
        )
    
    def _process_with_manual_composition(self, input_path: str, output_path: str) -> bool:
        """Process video using manual frame composition"""
        manual_composer = ManualComposer()
        return manual_composer.compose_video(
            input_path, output_path, self.face_detector, self.region_detector
        )
    
    def _default_config(self) -> dict:
        """Default pipeline configuration"""
        return {
            'min_face_conf': 0.3,
            'shot_threshold': 0.3,
            'face_alpha': 0.3,
            'slide_window': 8,
            'autoflip_config': None
        }
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            self.face_detector.close()
            self.autoflip_runner.cleanup()
            self.autoflip_hints.clear()
        except:
            pass

# Test complete pipeline
def test_complete_pipeline():
    """Test the complete reframing pipeline"""
    
    # Test configuration
    config = {
        'min_face_conf': 0.3,
        'shot_threshold': 0.3,
        'face_alpha': 0.3,
        'slide_window': 8
    }
    
    # Initialize pipeline
    pipeline = WebinarReframingPipeline(config)
    
    # Test processing
    input_video = "data/samples/s5.mp4"
    output_video = "output/reframed_test.mp4"
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Process video (try AutoFlip first, fallback to manual)
    success = pipeline.process_video(
        input_video, output_video, use_autoflip=True
    )
    
    if success:
        print("Pipeline test completed successfully!")
        print(f"Output saved to: {output_video}")
    else:
        print("Pipeline test failed")

if __name__ == "__main__":
    test_complete_pipeline()