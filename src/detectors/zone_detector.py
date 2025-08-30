import cv2
import numpy as np
from typing import Tuple, Optional, List

class UniversalWebinarDetector:
    """Universal detector that works regardless of face position"""
    
    def get_regions(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """
        Universal region detection that adapts to any face position
        """
        h, w = frame_bgr.shape[:2]
        
        if face_bbox is None:
            # No face: assume standard layout
            return self._get_default_regions(w, h)
        
        # Expand face region
        face_region = self._expand_face(face_bbox, w, h)
        
        # Find best slide area that doesn't overlap with face
        slide_region = self._find_best_slide_area(face_region, w, h)
        
        return face_region, slide_region
    
    def _expand_face(self, face_bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
        """Expand face to include head + shoulders"""
        x1, y1, x2, y2 = face_bbox
        
        # Calculate current dimensions
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Expand by 60% horizontally, 80% vertically
        expand_w = int(face_w * 0.6)
        expand_h = int(face_h * 0.8)
        
        # Apply expansion with bounds checking
        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(w, x2 + expand_w)
        new_y2 = min(h, y2 + expand_h)
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    def _find_best_slide_area(self, face_region: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Find largest rectangular area that doesn't overlap with face
        """
        fx1, fy1, fx2, fy2 = face_region
        
        # Define all possible slide areas
        candidate_areas = [
            # Four quadrants approach
            (0, 0, w//2, h//2),           # Top-left quadrant
            (w//2, 0, w, h//2),           # Top-right quadrant  
            (0, h//2, w//2, h),           # Bottom-left quadrant
            (w//2, h//2, w, h),           # Bottom-right quadrant
            
            # Half-screen approaches
            (0, 0, w//2, h),              # Left half
            (w//2, 0, w, h),              # Right half
            (0, 0, w, h//2),              # Top half
            (0, h//2, w, h),              # Bottom half
            
            # L-shaped regions (avoiding face area)
            (0, 0, fx1-20, h),            # Left of face
            (fx2+20, 0, w, h),            # Right of face
            (0, 0, w, fy1-20),            # Above face
            (0, fy2+20, w, h),            # Below face
            
            # Corner regions
            (0, 0, w*2//3, h*2//3),       # Large top-left
            (w//3, 0, w, h*2//3),         # Large top-right
            (0, h//3, w*2//3, h),         # Large bottom-left
            (w//3, h//3, w, h),           # Large bottom-right
        ]
        
        best_area = 0
        best_region = (0, 0, w//2, h)  # Fallback
        
        for x1, y1, x2, y2 in candidate_areas:
            # Ensure valid coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, max(x1 + 200, x2))  # Minimum 200px width
            y2 = min(h, max(y1 + 150, y2))  # Minimum 150px height
            
            candidate_region = (x1, y1, x2, y2)
            
            # Calculate overlap with face
            overlap_ratio = self._calculate_overlap_ratio(candidate_region, face_region)
            
            # Skip if too much overlap
            if overlap_ratio > 0.15:  # Allow max 15% overlap
                continue
            
            # Calculate area
            area = (x2 - x1) * (y2 - y1)
            
            # Must be reasonably large (at least 25% of frame)
            if area < 0.25 * w * h:
                continue
            
            # Score = area * (1 - overlap) * aspect_ratio_bonus
            aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
            aspect_bonus = 1.0
            if 1.2 <= aspect_ratio <= 2.5:  # Prefer landscape slides
                aspect_bonus = 1.3
            
            score = area * (1 - overlap_ratio) * aspect_bonus
            
            if score > best_area:
                best_area = score
                best_region = candidate_region
        
        return best_region
    
    def _calculate_overlap_ratio(self, region1: Tuple[int, int, int, int], region2: Tuple[int, int, int, int]) -> float:
        """Calculate what fraction of region1 overlaps with region2"""
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No overlap
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        region1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        return intersection_area / max(region1_area, 1)
    
    def _get_default_regions(self, w: int, h: int) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Default regions when no face is detected"""
        # Assume webcam in corner, main content elsewhere
        face_region = (0, 0, min(300, w//4), min(200, h//4))  # Small corner
        slide_region = (w//4, 0, w, h)  # Rest of screen
        
        return face_region, slide_region

# Test with different face positions
def test_universal_detector():
    """Test with various face positions"""
    detector = UniversalWebinarDetector()
    
    test_cases = [
        ((50, 50, 200, 200), "Top-left face"),
        ((1000, 50, 1200, 200), "Top-right face"), 
        ((50, 500, 200, 650), "Bottom-left face"),
        ((1000, 500, 1200, 650), "Bottom-right face"),
        ((500, 300, 700, 450), "Center face"),
        (None, "No face detected")
    ]
    
    for face_bbox, description in test_cases:
        face_region, slide_region = detector.get_regions(
            np.zeros((720, 1280, 3), dtype=np.uint8), face_bbox
        )
        
        print(f"\n{description}:")
        print(f"  Face: {face_region}")
        print(f"  Slide: {slide_region}")
        
        if face_bbox:
            overlap = detector._calculate_overlap_ratio(slide_region, face_region)
            print(f"  Overlap: {overlap:.1%}")

if __name__ == "__main__":
    test_universal_detector()