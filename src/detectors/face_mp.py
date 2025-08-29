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
        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=min_confidence
        )

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if not results.multi_face_landmarks:
            print("No faces detected")
            return None, 0.0

        # Get face landmarks and create bounding box
        face_landmarks = results.multi_face_landmarks[0]  # First face
        h, w = frame_bgr.shape[:2]

        # Get min/max coordinates from landmarks
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]

        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))

        confidence = 0.9  # Face Mesh doesn't give confidence, assume high
        print(f"Face detected at ({x1},{y1},{x2},{y2})")

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
    """Test face detection on actual video"""
    detector = FaceDetector(min_confidence=0.3)  # Lower confidence
    
    # Use actual video instead of dummy frame
    cap = cv2.VideoCapture("data/samples.mp4")  # Your video path
    
    if not cap.isOpened():
        print("Cannot open video - using webcam")
        cap = cv2.VideoCapture(0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        bbox, conf = detector.detect(frame)
        
        if bbox:
            print(f"Frame {frame_count}: Face found! Confidence: {conf:.2f}")
            break
        elif frame_count % 30 == 0:  # Print every 30 frames
            print(f"Frame {frame_count}: No face yet...")
        
        if frame_count > 300:  # Stop after 10 seconds at 30fps
            print("No face found in first 300 frames")
            break
    
    cap.release()
    detector.close()

if __name__ == "__main__":
    test_face_detection()