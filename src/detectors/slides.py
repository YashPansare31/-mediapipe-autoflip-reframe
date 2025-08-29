import cv2
import numpy as np
from typing import Optional, Tuple, List

class SlideDetector:
    def __init__(self, min_area_ratio: float = 0.1, aspect_ratio_range: Tuple[float, float] = (0.7, 2.2)):
        """
        Initialize slide detection
        
        Args:
            min_area_ratio: Minimum area as fraction of frame (0.1 = 10%)
            aspect_ratio_range: (min_ar, max_ar) acceptable aspect ratios
        """
        self.min_area_ratio = min_area_ratio
        self.min_aspect_ratio, self.max_aspect_ratio = aspect_ratio_range
    
    def find_slide_roi(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Find ONLY the slide/presentation content area, excluding face region
        """
        h, w = frame_bgr.shape[:2]

        # Step 1: Create mask to exclude face area completely
        mask = np.ones_like(frame_bgr[:,:,0], dtype=np.uint8) * 255

        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            # Expand face exclusion zone significantly
            face_w, face_h = fx2 - fx1, fy2 - fy1
            expand_x = int(face_w * 0.8)  # 80% expansion
            expand_y = int(face_h * 0.8)  # 80% expansion

            excl_x1 = max(0, fx1 - expand_x)
            excl_y1 = max(0, fy1 - expand_y)
            excl_x2 = min(w, fx2 + expand_x)
            excl_y2 = min(h, fy2 + expand_y)

            # Black out entire face region
            mask[excl_y1:excl_y2, excl_x1:excl_x2] = 0
            print(f"Excluding face region: ({excl_x1},{excl_y1},{excl_x2},{excl_y2})")

        # Step 2: Look for slide content in remaining areas
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, mask)

        # Method: Find the largest bright rectangular region in non-face area
        bright_thresh = cv2.threshold(masked_gray, 180, 255, cv2.THRESH_BINARY)[1]

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bright_thresh = cv2.morphologyEx(bright_thresh, cv2.MORPH_CLOSE, kernel)
        bright_thresh = cv2.morphologyEx(bright_thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = 0
        min_area = 0.08 * w * h  # At least 8% of total frame

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            if area < min_area:
                continue
            
            # Must be reasonably rectangular (slide content)
            contour_area = cv2.contourArea(contour)
            rectangularity = contour_area / max(area, 1)
            if rectangularity < 0.6:
                continue
            
            # Aspect ratio should be landscape-ish for slides
            aspect_ratio = cw / max(ch, 1)
            if not (0.8 <= aspect_ratio <= 4.0):
                continue
            
            # Double-check no overlap with face
            if face_bbox is not None:
                overlap = self._calculate_overlap((x, y, x+cw, y+ch), face_bbox)
                if overlap > 0.1:  # Very strict - no overlap allowed
                    continue
                
            # Score: prefer larger, more rectangular regions
            score = area * rectangularity * aspect_ratio

            if score > best_score:
                best_score = score
                best_candidate = (x, y, x + cw, y + ch)

        # Fallback: If no good bright regions, try predefined slide areas
        if best_candidate is None:
            print("No slide content found, trying predefined areas...")
            return self.try_predefined_slide_areas(frame_bgr, face_bbox)

        return best_candidate

    def  try_predefined_slide_areas(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
            """
            Try common slide locations in webinar layouts
            """
            h, w = frame_bgr.shape[:2]

            # Common webinar slide positions
            slide_areas = [
                (0, 0, int(w*0.65), h),                    # Left 65% (most common)
                (0, 0, w, int(h*0.75)),                    # Top 75% 
                (int(w*0.1), int(h*0.1), int(w*0.7), int(h*0.8)),  # Center-left area
                (0, int(h*0.05), int(w*0.6), int(h*0.95)), # Left column avoiding top/bottom
            ]

            best_candidate = None
            best_score = 0

            for x1, y1, x2, y2 in slide_areas:
                # Skip if overlaps significantly with face
                if face_bbox is not None:
                    overlap = self._calculate_overlap((x1, y1, x2, y2), face_bbox)
                    if overlap > 0.25:
                        continue
                    
                area = (x2 - x1) * (y2 - y1)

                # Check if this area actually contains slide-like content
                roi = frame_bgr[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Look for text-like patterns in this region
                edges = cv2.Canny(gray_roi, 50, 150)
                edge_density = edges.mean()

                # Look for bright content (typical of slides)
                brightness = gray_roi.mean()

                # Combined score
                content_score = edge_density * brightness * area

                if content_score > best_score:
                    best_score = content_score
                    best_candidate = (x1, y1, x2, y2)

            return best_candidate   

    def _geometric_slide_detection(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Fallback: Assume slide is largest rectangular area not occupied by face
        """
        h, w = frame_bgr.shape[:2]

        # Divide frame into potential slide regions
        candidates = [
            (0, 0, w//2, h),           # Left half
            (w//3, 0, w, h//2),        # Right top
            (0, h//3, w, h),           # Bottom portion
            (w//4, h//4, 3*w//4, 3*h//4),  # Center region
        ]

        best_candidate = None
        best_score = 0

        for x1, y1, x2, y2 in candidates:
            # Skip if overlaps with face significantly
            if face_bbox is not None:
                overlap = self._calculate_overlap((x1, y1, x2, y2), face_bbox)
                if overlap > 0.3:
                    continue
                
            area = (x2 - x1) * (y2 - y1)
            aspect_ratio = (x2 - x1) / max(y2 - y1, 1)

            # Prefer landscape regions
            if 1.2 <= aspect_ratio <= 2.5:
                score = area * aspect_ratio

                # Prefer left/center regions
                center_x = (x1 + x2) // 2
                if center_x < w * 0.6:
                    score *= 1.3

                if score > best_score:
                    best_score = score
                    best_candidate = (x1, y1, x2, y2)

        return best_candidate
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap percentage between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)

        return intersection / area1
    
# Test function
def test_slide_detection():
    """Test slide detection on actual video"""
    detector = SlideDetector(min_area_ratio=0.05)  # Lower area requirement
    
    # Use actual video
    cap = cv2.VideoCapture("data/samples/s4.mp4")  # Your video path
    
    if not cap.isOpened():
        print("Cannot open video")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        bbox = detector.find_slide_roi(frame)
        
        if bbox:
            print(f"Frame {frame_count}: Slide found! {bbox}")
            # Show the frame
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Slide Detection', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            break
        elif frame_count % 30 == 0:
            print(f"Frame {frame_count}: No slide yet...")
            
        if frame_count > 300:
            print("No slide found in first 300 frames")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_slide_detection()