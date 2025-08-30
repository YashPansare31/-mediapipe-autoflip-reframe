import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

class ShotChangeDetector:
    """Detects scene cuts and shot changes in video"""
    
    def __init__(self, threshold: float = 0.3, history_size: int = 5):
        """
        Args:
            threshold: Color histogram change threshold (0-1)
            history_size: Number of frames to consider for stability
        """
        self.threshold = threshold
        self.history_size = history_size
        self.hist_history: deque = deque(maxlen=history_size)
        self.last_change_frame = -10  # Prevent rapid changes
    
    def detect_shot_change(self, frame: np.ndarray, frame_number: int) -> bool:
        """
        Detect if current frame represents a shot change
        
        Args:
            frame: Current video frame
            frame_number: Frame index
            
        Returns:
            True if shot change detected
        """
        # Calculate color histogram
        hist = self._calculate_histogram(frame)
        
        if len(self.hist_history) == 0:
            self.hist_history.append(hist)
            return False
        
        # Compare with recent history
        change_detected = False
        max_change = 0.0
        
        for prev_hist in self.hist_history:
            correlation = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
            change = 1.0 - correlation
            max_change = max(max_change, change)
            
            if change > self.threshold:
                # Prevent rapid consecutive detections
                if frame_number - self.last_change_frame > 10:
                    change_detected = True
                    self.last_change_frame = frame_number
                break
        
        self.hist_history.append(hist)
        
        if change_detected:
            print(f"Shot change detected at frame {frame_number} (change: {max_change:.3f})")
        
        return change_detected
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate normalized color histogram for frame"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for H and S channels
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # Normalize
        cv2.normalize(hist, hist)
        
        return hist

class ConflictResolver:
    """Resolves conflicts between face and slide regions"""
    
    def __init__(self, min_face_size: float = 0.02, min_slide_size: float = 0.15):
        """
        Args:
            min_face_size: Minimum face region as fraction of frame
            min_slide_size: Minimum slide region as fraction of frame
        """
        self.min_face_size = min_face_size
        self.min_slide_size = min_slide_size
    
    def resolve_conflicts(self, face_region: Optional[Tuple[int, int, int, int]], 
                         slide_region: Optional[Tuple[int, int, int, int]], 
                         frame_shape: Tuple[int, int]) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """
        Resolve overlaps and conflicts between regions
        
        Args:
            face_region: Face bounding box (x1, y1, x2, y2)
            slide_region: Slide bounding box (x1, y1, x2, y2)  
            frame_shape: (height, width) of frame
            
        Returns:
            (resolved_face_region, resolved_slide_region)
        """
        if face_region is None or slide_region is None:
            return face_region, slide_region
        
        h, w = frame_shape
        frame_area = w * h
        
        # Calculate overlap
        overlap_area = self._calculate_overlap_area(face_region, slide_region)
        overlap_ratio = overlap_area / frame_area
        
        # If minimal overlap, keep as is
        if overlap_ratio < 0.05:
            return face_region, slide_region
        
        print(f"Resolving region conflict (overlap: {overlap_ratio:.1%})")
        
        # Priority: Face visibility > slide size
        face_area = self._calculate_area(face_region)
        slide_area = self._calculate_area(slide_region)
        
        # If face is very small, prioritize slide
        if face_area < self.min_face_size * frame_area:
            return None, slide_region
        
        # If slide would become too small, prioritize face
        adjusted_slide = self._adjust_slide_for_face(slide_region, face_region, frame_shape)
        adjusted_slide_area = self._calculate_area(adjusted_slide) if adjusted_slide else 0
        
        if adjusted_slide_area < self.min_slide_size * frame_area:
            # Shrink face region instead
            adjusted_face = self._shrink_face_region(face_region, 0.9)
            return adjusted_face, slide_region
        
        return face_region, adjusted_slide
    
    def _calculate_overlap_area(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> int:
        """Calculate overlap area between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0
        
        return (x2_i - x1_i) * (y2_i - y1_i)
    
    def _calculate_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """Calculate area of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _adjust_slide_for_face(self, slide_region: Tuple[int, int, int, int],
                              face_region: Tuple[int, int, int, int], 
                              frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Adjust slide region to avoid face"""
        sx1, sy1, sx2, sy2 = slide_region
        fx1, fy1, fx2, fy2 = face_region
        h, w = frame_shape
        
        # Try to move slide region away from face
        face_center_x = (fx1 + fx2) // 2
        slide_center_x = (sx1 + sx2) // 2
        
        if face_center_x < w // 2:  # Face on left, move slide right
            new_sx1 = max(fx2 + 10, sx1)
            new_sx2 = min(w, new_sx1 + (sx2 - sx1))
            if new_sx2 - new_sx1 > 100:  # Ensure minimum width
                return (new_sx1, sy1, new_sx2, sy2)
        else:  # Face on right, move slide left
            new_sx2 = min(fx1 - 10, sx2)
            new_sx1 = max(0, new_sx2 - (sx2 - sx1))
            if new_sx2 - new_sx1 > 100:
                return (new_sx1, sy1, new_sx2, sy2)
        
        # If horizontal adjustment failed, try vertical
        face_center_y = (fy1 + fy2) // 2
        if face_center_y < h // 2:  # Face on top, move slide down
            new_sy1 = max(fy2 + 10, sy1)
            new_sy2 = min(h, new_sy1 + (sy2 - sy1))
            if new_sy2 - new_sy1 > 100:
                return (sx1, new_sy1, sx2, new_sy2)
        
        return None  # Couldn't adjust
    
    def _shrink_face_region(self, face_region: Tuple[int, int, int, int], 
                           factor: float) -> Tuple[int, int, int, int]:
        """Shrink face region by given factor"""
        x1, y1, x2, y2 = face_region
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        new_w = int((x2 - x1) * factor) // 2
        new_h = int((y2 - y1) * factor) // 2
        
        return (center_x - new_w, center_y - new_h, 
                center_x + new_w, center_y + new_h)

# Test shot detection and conflict resolution
def test_shot_and_conflict():
    """Test shot change detection and conflict resolution"""
    shot_detector = ShotChangeDetector(threshold=0.3)
    conflict_resolver = ConflictResolver()
    
    cap = cv2.VideoCapture("data/samples/s5.mp4")
    
    frame_count = 0
    shot_changes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Test shot detection
        is_shot_change = shot_detector.detect_shot_change(frame, frame_count)
        if is_shot_change:
            shot_changes.append(frame_count)
        
        # Test conflict resolution with mock regions
        if frame_count % 30 == 0:
            # Mock overlapping regions
            face_region = (400, 200, 600, 400)
            slide_region = (350, 150, 800, 450)
            
            resolved_face, resolved_slide = conflict_resolver.resolve_conflicts(
                face_region, slide_region, frame.shape[:2]
            )
            
            print(f"Frame {frame_count}: Original face {face_region}, slide {slide_region}")
            print(f"Frame {frame_count}: Resolved face {resolved_face}, slide {resolved_slide}")
        
        if frame_count > 300:
            break
    
    cap.release()
    
    print(f"\nShot changes detected at frames: {shot_changes}")
    print(f"Total shot changes: {len(shot_changes)}")

if __name__ == "__main__":
    test_shot_and_conflict()