import cv2
import numpy as np
from typing import Optional, Tuple, List

class SlideDetector:
    def __init__(self, min_area_ratio: float = 0.1, aspect_ratio_range: Tuple[float, float] = (0.7, 2.2)):
        """
        Initialize multi-method slide detection
        
        Args:
            min_area_ratio: Minimum area as fraction of frame (0.1 = 10%)
            aspect_ratio_range: (min_ar, max_ar) acceptable aspect ratios
        """
        self.min_area_ratio = min_area_ratio
        self.min_aspect_ratio, self.max_aspect_ratio = aspect_ratio_range
    
    def find_slide_roi(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Multi-method slide detection with fallbacks
        """
        # Method 1: Bright region detection
        result = self._detect_bright_regions(frame_bgr, face_bbox)
        if result is not None:
            return result
        
        # Method 2: Edge-based detection  
        result = self._detect_edge_regions(frame_bgr, face_bbox)
        if result is not None:
            return result
        
        # Method 3: Zone-based fallback
        result = self._detect_zone_based(frame_bgr, face_bbox)
        return result
    
    def _detect_bright_regions(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """Method 1: Detect bright rectangular regions (typical slides)"""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Find bright regions
        bright_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_region = None
        best_score = 0
        min_area = self.min_area_ratio * w * h
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            
            if area < min_area:
                continue
            
            # Rectangularity check
            contour_area = cv2.contourArea(contour)
            rectangularity = contour_area / max(area, 1)
            if rectangularity < 0.6:
                continue
            
            # Aspect ratio check
            aspect_ratio = cw / max(ch, 1)
            if not (0.8 <= aspect_ratio <= 4.0):
                continue
            
            # Face overlap check
            if face_bbox is not None:
                overlap = self._calculate_overlap((x, y, x+cw, y+ch), face_bbox)
                if overlap > 0.1:
                    continue
            
            # Content validation
            roi = frame_bgr[y:y+ch, x:x+cw]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            brightness = roi_gray.mean()
            contrast = roi_gray.std()
            
            if brightness < 160 or contrast < 15:
                continue
            
            score = area * rectangularity * brightness
            if score > best_score:
                best_score = score
                best_region = (x, y, x + cw, y + ch)
        
        if best_region:
            print("Bright region detection method succeeded")
        return best_region
    
    def _detect_edge_regions(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """Method 2: Detect regions with high edge density (text-rich areas)"""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create edge density map
        kernel = np.ones((30, 30), np.float32) / 900
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
        
        # Threshold high density areas
        density_thresh = edge_density > (edge_density.max() * 0.3)
        density_thresh = density_thresh.astype(np.uint8) * 255
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        density_thresh = cv2.morphologyEx(density_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(density_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_region = None
        best_score = 0
        min_area = self.min_area_ratio * w * h
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            
            if area < min_area:
                continue
            
            aspect_ratio = cw / max(ch, 1)
            if not (0.7 <= aspect_ratio <= 3.5):
                continue
            
            # Face overlap check
            if face_bbox is not None:
                overlap = self._calculate_overlap((x, y, x+cw, y+ch), face_bbox)
                if overlap > 0.15:
                    continue
            
            # Calculate edge density in this region
            roi_edges = edges[y:y+ch, x:x+cw]
            density_score = roi_edges.mean()
            
            score = area * density_score * aspect_ratio
            if score > best_score:
                best_score = score
                best_region = (x, y, x + cw, y + ch)
        
        if best_region:
            print("Edge-based detection method succeeded")
        return best_region
    
    def _detect_zone_based(self, frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """Method 3: Zone-based fallback (geometric approach)"""
        h, w = frame_bgr.shape[:2]
        
        # Define candidate zones
        candidates = [
            (0, 0, w//2, h),              # Left half
            (w//3, 0, w, h//2),           # Top-right
            (0, h//4, w*3//4, h),         # Bottom-left large
            (w//4, 0, w, h*3//4),         # Top-right large
            (0, 0, w*2//3, h*2//3),       # Top-left large
        ]
        
        best_region = None
        best_score = 0
        
        for x1, y1, x2, y2 in candidates:
            # Ensure minimum size
            if (x2 - x1) * (y2 - y1) < self.min_area_ratio * w * h:
                continue
            
            # Face overlap check
            if face_bbox is not None:
                overlap = self._calculate_overlap((x1, y1, x2, y2), face_bbox)
                if overlap > 0.25:
                    continue
            
            # Analyze region content
            roi = frame_bgr[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Look for slide-like characteristics
            brightness = roi_gray.mean()
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_density = edges.mean()
            
            # Prefer brighter regions with some edge content
            area = (x2 - x1) * (y2 - y1)
            score = area * brightness * (1 + edge_density)
            
            if score > best_score:
                best_score = score
                best_region = (x1, y1, x2, y2)
        
        if best_region:
            print("Zone-based fallback method succeeded")
        return best_region
    
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
        
        return intersection / max(area1, 1)

# Test all detection methods
def test_slide_detection():
    """Test slide detection with method diagnostics"""
    detector = SlideDetector(min_area_ratio=0.05)
    
    cap = cv2.VideoCapture("data/samples/s5.mp4")
    
    if not cap.isOpened():
        print("Cannot open video")
        return
    
    frame_count = 0
    detection_stats = {"bright": 0, "edge": 0, "zone": 0, "none": 0}
    
    print("Testing slide detection methods...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Test each method individually
        method1 = detector._detect_bright_regions(frame)
        method2 = detector._detect_edge_regions(frame) if method1 is None else None
        method3 = detector._detect_zone_based(frame) if method1 is None and method2 is None else None
        
        # Combined detection
        final_bbox = detector.find_slide_roi(frame)
        
        # Track which method worked
        if method1 is not None:
            detection_stats["bright"] += 1
        elif method2 is not None:
            detection_stats["edge"] += 1
        elif method3 is not None:
            detection_stats["zone"] += 1
        else:
            detection_stats["none"] += 1
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: M1={method1 is not None}, M2={method2 is not None}, M3={method3 is not None}")
            
            if final_bbox:
                x1, y1, x2, y2 = final_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, 'SLIDE', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Slide Detection', frame)
                cv2.waitKey(1)
        
        if frame_count > 300:  # Test first 10 seconds
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nSlide Detection Results over {frame_count} frames:")
    for method, count in detection_stats.items():
        percentage = (count / frame_count) * 100
        print(f"  {method}: {count} frames ({percentage:.1f}%)")

if __name__ == "__main__":
    test_slide_detection()