import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

class FaceDetector:
    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize MediaPipe Face Detection
        
        Args:
            min_confidence: Minimum detection confidence threshold
        """
        self.min_confidence = min_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_confidence
        )
    
    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Detect largest face in frame
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            ((x1, y1, x2, y2), confidence) or (None, 0.0) if no face found
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.detector.process(rgb_frame)
        
        if not results.detections:
            return None, 0.0
        
        # Get frame dimensions
        h, w = frame_bgr.shape[:2]
        
        # Find largest detection (assume main speaker)
        best_detection = max(results.detections, 
                           key=lambda d: (d.location_data.relative_bounding_box.width * 
                                        d.location_data.relative_bounding_box.height))
        
        # Extract bounding box
        bbox = best_detection.location_data.relative_bounding_box
        confidence = best_detection.score[0]
        
        # Convert relative coordinates to absolute pixels
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h) 
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        return (x1, y1, x2, y2), confidence
    
    def expand_face_bbox(self, bbox: Tuple[int, int, int, int], 
                        frame_shape: Tuple[int, int], 
                        h_expand: float = 0.2, v_expand: float = 0.35) -> Tuple[int, int, int, int]:
        """
        Expand face bounding box to include head + upper torso
        
        Args:
            bbox: (x1, y1, x2, y2) face bounding box
            frame_shape: (height, width) of frame
            h_expand: Horizontal expansion ratio (0.2 = 20%)
            v_expand: Vertical expansion ratio (0.35 = 35%)
            
        Returns:
            Expanded (x1, y1, x2, y2) bounding box
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape
        
        # Calculate expansion
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        h_pad = int(bbox_w * h_expand)
        v_pad = int(bbox_h * v_expand)
        
        # Expand and clamp to frame boundaries
        x1_exp = max(0, x1 - h_pad)
        y1_exp = max(0, y1 - v_pad)
        x2_exp = min(w, x2 + h_pad)
        y2_exp = min(h, y2 + v_pad)
        
        return (x1_exp, y1_exp, x2_exp, y2_exp)
    
    def close(self):
        """Clean up MediaPipe resources"""
        self.detector.close()

# Test function
def test_face_detection():
    """Test face detection on a sample image/frame"""
    detector = FaceDetector(min_confidence=0.6)
    
    # Create a dummy frame for testing
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox, conf = detector.detect(test_frame)
    print(f"Detection result: {bbox}, confidence: {conf}")
    
    detector.close()

if __name__ == "__main__":
    test_face_detection()