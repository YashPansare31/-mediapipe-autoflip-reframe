import cv2
import sys
sys.path.append('src')

from src.detectors.face_mp import FaceDetector
from src.detectors.slides import SlideDetector
from src.trackers.temporal import EMATracker, TemporalMedianTracker
from src.layout.composer import LayoutComposer
from src.autoflip.hints import AutoFlipHints

def test_day2_integration():
    """Test all Day 2 components working together"""
    
    # Initialize components
    face_detector = FaceDetector(min_confidence=0.3)
    slide_detector = SlideDetector(min_area_ratio=0.05)
    face_tracker = EMATracker(alpha=0.3)
    slide_tracker = TemporalMedianTracker(window_size=8)
    composer = LayoutComposer()
    hints = AutoFlipHints()
    
    # Test video
    video_path = "data/samples/s5.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {width}x{height} @ {fps} fps")
    
    frame_count = 0
    
    # Create output video writer for composed frames
    out_writer = cv2.VideoWriter('day2_test_output.mp4', 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, (1080, 1920))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Skip early frames
            if frame_count < 50:
                continue
            
            # Detect
            face_bbox, face_conf = face_detector.detect(frame)
            slide_bbox = slide_detector.find_slide_roi(frame, face_bbox)
            
            # Smooth/track
            stable_face = face_tracker.update(face_bbox, face_conf)
            stable_slide = slide_tracker.update(slide_bbox)
            
            # Compose layout
            composed_frame = composer.compose_frame(frame, stable_face, stable_slide)
            
            # Add to AutoFlip hints
            hints.add_frame_detections(frame_count, timestamp, stable_face, face_conf,
                                     stable_slide, width, height)
            
            # Write output
            out_writer.write(composed_frame)
            
            # Show progress
            if frame_count % 30 == 0:
                print(f"Processed frame {frame_count}, face: {stable_face is not None}, slide: {stable_slide is not None}")
            
            # Process limited frames for testing
            if frame_count > 9000:  # ~10 seconds
                break
                
    except KeyboardInterrupt:
        print("Processing interrupted")
    
    finally:
        # Cleanup
        cap.release()
        out_writer.release()
        face_detector.close()
        
        # Save hints
        hints.save_hints()
        
        print(f"\nDay 2 Integration Test Complete!")
        print(f"- Processed {frame_count} frames")
        print(f"- Output: day2_test_output.mp4")
        print(f"- AutoFlip hints saved")

if __name__ == "__main__":
    test_day2_integration()