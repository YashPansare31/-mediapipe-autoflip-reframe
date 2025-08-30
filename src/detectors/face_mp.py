import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

class FaceDetector:
    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize with BOTH Face Detection AND Face Mesh for robustness
        """
        self.min_confidence = min_confidence
        
        # Primary: Face Detection (more robust for videos)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full range model (better for videos)
            min_detection_confidence=min_confidence
        )
        
        # Backup: Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,  # Allow multiple faces
            refine_landmarks=False,
            min_detection_confidence=min_confidence * 0.7  # Lower threshold for backup
        )
        
        # Fallback: Haar Cascade
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Multi-method face detection with fallbacks
        """
        # Method 1: MediaPipe Face Detection
        result = self._detect_with_face_detection(frame_bgr)
        if result[0] is not None:
            return result
        
        # Method 2: MediaPipe Face Mesh  
        result = self._detect_with_face_mesh(frame_bgr)
        if result[0] is not None:
            return result
        
        # Method 3: Haar Cascade (fallback)
        result = self._detect_with_haar(frame_bgr)
        return result
    
    def _detect_with_face_detection(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Primary method: MediaPipe Face Detection"""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        if not results.detections:
            return None, 0.0
        
        h, w = frame_bgr.shape[:2]
        
        # Find largest/most confident detection
        best_detection = max(results.detections, 
                           key=lambda d: d.score[0] * (d.location_data.relative_bounding_box.width * 
                                                      d.location_data.relative_bounding_box.height))
        
        bbox = best_detection.location_data.relative_bounding_box
        confidence = best_detection.score[0]
        
        # Convert to absolute coordinates
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        print(f"Face Detection method: confidence {confidence:.2f}")
        return (x1, y1, x2, y2), confidence
    
    def _detect_with_face_mesh(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Backup method: MediaPipe Face Mesh"""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, 0.0
        
        h, w = frame_bgr.shape[:2]
        
        # Use first (largest) face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get bounding box from landmarks
        x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
        
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        print(f"Face Mesh method: estimated confidence 0.8")
        return (x1, y1, x2, y2), 0.8
    
    def _detect_with_haar(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Fallback method: OpenCV Haar Cascade"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with multiple scales
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, 0.0
        
        # Get largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])  # width * height
        x, y, w_face, h_face = largest_face
        
        print(f"Haar Cascade method: estimated confidence 0.6")
        return (x, y, x + w_face, y + h_face), 0.6
    
    def close(self):
        """Clean up resources"""
        self.face_detector.close()
        self.face_mesh.close()
    
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

# Test all detection methods
def test_multi_method_detection():
    """Test all face detection methods"""
    detector = FaceDetector(min_confidence=0.4)  # Lower threshold
    
    video_path = "data/sample.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Cannot open video")
        return
    
    detection_counts = {"face_detection": 0, "face_mesh": 0, "haar": 0, "none": 0}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Test each method individually to see which works
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Method 1 test
        results1 = detector.face_detector.process(rgb_frame)
        method1_works = results1.detections is not None and len(results1.detections) > 0
        
        # Method 2 test  
        results2 = detector.face_mesh.process(rgb_frame)
        method2_works = results2.multi_face_landmarks is not None and len(results2.multi_face_landmarks) > 0
        
        # Method 3 test
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces3 = detector.haar_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        method3_works = len(faces3) > 0
        
        # Combined detection
        bbox, conf = detector.detect(frame)
        
        # Track which method worked
        if bbox is not None:
            if method1_works:
                detection_counts["face_detection"] += 1
            elif method2_works:
                detection_counts["face_mesh"] += 1
            elif method3_works:
                detection_counts["haar"] += 1
        else:
            detection_counts["none"] += 1
        
        # Visualize every 30th frame
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: M1={method1_works}, M2={method2_works}, M3={method3_works}, Final={bbox is not None}")
            
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Multi-Method Face Detection', frame)
            cv2.waitKey(1)
        
        if frame_count > 300:  # Test first 10 seconds
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print(f"\nDetection Results over {frame_count} frames:")
    for method, count in detection_counts.items():
        percentage = (count / frame_count) * 100
        print(f"  {method}: {count} frames ({percentage:.1f}%)")

if __name__ == "__main__":
    test_multi_method_detection()