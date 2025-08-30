import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
import os

from detectors.face_mp import FaceDetector

class EdgeCaseHandler:
    """Handles edge cases and failure modes"""
    
    def __init__(self):
        self.failure_modes = {
            'no_face_long_term': 0,
            'no_slide_long_term': 0,
            'rapid_movement': 0,
            'poor_quality_frames': 0,
            'aspect_ratio_issues': 0
        }
    
    def handle_no_face_detection(self, frame: np.ndarray, frame_number: int, 
                                last_face_frame: int) -> Optional[Tuple[int, int, int, int]]:
        """Handle periods with no face detection"""
        
        frames_without_face = frame_number - last_face_frame
        
        if frames_without_face > 90:  # 3 seconds at 30fps
            self.failure_modes['no_face_long_term'] += 1
            
            # Strategy 1: Look for face with very low confidence
            face_detector_emergency = FaceDetector(min_confidence=0.1)
            emergency_bbox, emergency_conf = face_detector_emergency.detect(frame)
            face_detector_emergency.close()
            
            if emergency_bbox:
                print(f"Emergency face detection at frame {frame_number} (conf: {emergency_conf:.2f})")
                return emergency_bbox
            
            # Strategy 2: Use motion detection to find active speaker
            motion_region = self._detect_motion_region(frame)
            if motion_region:
                print(f"Using motion-based speaker region at frame {frame_number}")
                return motion_region
            
            # Strategy 3: Use default speaker position based on layout
            default_region = self._get_default_speaker_region(frame.shape[:2])
            print(f"Using default speaker region at frame {frame_number}")
            return default_region
        
        return None
    
    def handle_no_slide_detection(self, frame: np.ndarray, frame_number: int,
                                 face_region: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Handle periods with no slide detection"""
        
        h, w = frame.shape[:2]
        
        # Strategy 1: Use complementary region to face
        if face_region:
            return self._get_complementary_slide_region(face_region, w, h)
        
        # Strategy 2: Use default slide layout
        return self._get_default_slide_region(w, h)
    
    def handle_rapid_movement(self, current_bbox: Tuple[int, int, int, int],
                            previous_bbox: Tuple[int, int, int, int],
                            max_movement: int = 100) -> Tuple[int, int, int, int]:
        """Handle rapid/jerky movement by limiting displacement"""
        
        if previous_bbox is None:
            return current_bbox
        
        # Calculate movement
        curr_center = self._get_center(current_bbox)
        prev_center = self._get_center(previous_bbox)
        movement = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
        
        if movement > max_movement:
            self.failure_modes['rapid_movement'] += 1
            
            # Limit movement
            direction = np.array(curr_center) - np.array(prev_center)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                limited_center = np.array(prev_center) + direction * max_movement
                
                # Reconstruct bbox around limited center
                cx1, cy1, cx2, cy2 = current_bbox
                width, height = cx2 - cx1, cy2 - cy1
                
                new_x1 = int(limited_center[0] - width // 2)
                new_y1 = int(limited_center[1] - height // 2)
                new_x2 = new_x1 + width
                new_y2 = new_y1 + height
                
                return (new_x1, new_y1, new_x2, new_y2)
        
        return current_bbox
    
    def handle_poor_quality_frame(self, frame: np.ndarray) -> np.ndarray:
        """Improve quality of poor frames"""
        
        # Check frame quality metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness check
        brightness = gray.mean()
        if brightness < 50:  # Very dark
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
            self.failure_modes['poor_quality_frames'] += 1
        
        # Blur check
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Very blurry
            # Mild sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel * 0.1)
            self.failure_modes['poor_quality_frames'] += 1
        
        return frame
    
    def handle_aspect_ratio_issues(self, bbox: Tuple[int, int, int, int], 
                                 frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Fix problematic aspect ratios"""
        
        x1, y1, x2, y2 = bbox
        h, w = frame_shape
        
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return (0, 0, 100, 100)  # Emergency fallback
        
        aspect_ratio = width / height
        
        # Fix extreme aspect ratios
        if aspect_ratio > 3.0:  # Too wide
            self.failure_modes['aspect_ratio_issues'] += 1
            # Make it more square
            target_width = int(height * 2.0)
            center_x = (x1 + x2) // 2
            new_x1 = max(0, center_x - target_width // 2)
            new_x2 = min(w, new_x1 + target_width)
            return (new_x1, y1, new_x2, y2)
        
        elif aspect_ratio < 0.5:  # Too tall
            self.failure_modes['aspect_ratio_issues'] += 1
            # Make it wider
            target_height = int(width * 1.5)
            center_y = (y1 + y2) // 2
            new_y1 = max(0, center_y - target_height // 2)
            new_y2 = min(h, new_y1 + target_height)
            return (x1, new_y1, x2, new_y2)
        
        return bbox
    
    def _detect_motion_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect active regions using motion (placeholder for future implementation)"""
        # This would require frame differencing - simplified for now
        h, w = frame.shape[:2]
        
        # Return likely speaker region (typically corners in webinars)
        return (0, 0, min(400, w//3), min(300, h//3))
    
    def _get_default_speaker_region(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Default speaker region when all detection fails"""
        h, w = frame_shape
        # Assume speaker in corner (common webinar pattern)
        return (0, 0, min(350, w//4), min(250, h//4))
    
    def _get_complementary_slide_region(self, face_region: Tuple[int, int, int, int],
                                      frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Get slide region complementary to face region"""
        fx1, fy1, fx2, fy2 = face_region
        
        # If face is on left, slide on right
        if fx2 < frame_w // 2:
            return (fx2 + 20, 0, frame_w, frame_h)
        
        # If face is on right, slide on left
        elif fx1 > frame_w // 2:
            return (0, 0, fx1 - 20, frame_h)
        
        # If face is on top, slide on bottom
        elif fy2 < frame_h // 2:
            return (0, fy2 + 20, frame_w, frame_h)
        
        # Default: left side
        return (0, 0, frame_w // 2, frame_h)
    
    def _get_default_slide_region(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Default slide region"""
        # Assume slide takes main area (right side or center)
        return (frame_w // 3, 0, frame_w, frame_h)
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _default_config(self) -> Dict:
        return {
            'min_face_conf': 0.3,
            'shot_threshold': 0.3,
            'face_alpha': 0.3,
            'slide_window': 8,
            'target_bitrate': '8M'
        }
    
    def _cleanup(self):
        """Clean up resources"""
        self.face_detector.close()
        self.video_encoder.cleanup()
    
    def get_failure_report(self) -> Dict:
        """Get report of handled failure modes"""
        return self.failure_modes.copy()

# Test edge case handling
def test_edge_cases():
    """Test edge case handling"""
    from src.detectors.face_mp import FaceDetector
    
    handler = EdgeCaseHandler()
    
    # Test 1: No face detection
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = handler.handle_no_face_detection(test_frame, 100, 10)
    print(f"No face handling: {result}")
    
    # Test 2: Rapid movement
    bbox1 = (100, 100, 200, 200)
    bbox2 = (300, 300, 400, 400)  # Large movement
    limited = handler.handle_rapid_movement(bbox2, bbox1, max_movement=50)
    print(f"Movement limiting: {bbox2} -> {limited}")
    
    # Test 3: Aspect ratio fixing
    bad_bbox = (100, 100, 500, 120)  # Very wide
    fixed = handler.handle_aspect_ratio_issues(bad_bbox, (720, 1280))
    print(f"Aspect ratio fix: {bad_bbox} -> {fixed}")
    
    print("Edge case testing complete")

if __name__ == "__main__":
    test_edge_cases()