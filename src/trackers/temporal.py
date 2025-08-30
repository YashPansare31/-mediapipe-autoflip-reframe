import numpy as np
from typing import Optional, Tuple, List, Deque
from collections import deque

class EMATracker:
    """Exponential Moving Average tracker for smooth bounding boxes"""
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0 = no update, 1 = no smoothing)
        """
        self.alpha = alpha
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.confidence_history: Deque[float] = deque(maxlen=10)
    
    def update(self, bbox: Optional[Tuple[int, int, int, int]], confidence: float = 1.0) -> Optional[Tuple[int, int, int, int]]:
        """
        Update tracker with new detection
        
        Args:
            bbox: New bounding box (x1, y1, x2, y2) or None
            confidence: Detection confidence
            
        Returns:
            Smoothed bounding box or None
        """
        self.confidence_history.append(confidence)
        
        if bbox is None:
            # No detection - keep previous if confidence history is good
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            if avg_confidence > 0.5 and self.current_bbox is not None:
                return self.current_bbox  # Hold previous detection
            else:
                self.current_bbox = None
                return None
        
        if self.current_bbox is None:
            # First detection
            self.current_bbox = bbox
            return bbox
        
        # Smooth with EMA
        x1, y1, x2, y2 = bbox
        cx1, cy1, cx2, cy2 = self.current_bbox
        
        # Apply exponential moving average
        smooth_x1 = int(self.alpha * x1 + (1 - self.alpha) * cx1)
        smooth_y1 = int(self.alpha * y1 + (1 - self.alpha) * cy1)
        smooth_x2 = int(self.alpha * x2 + (1 - self.alpha) * cx2)
        smooth_y2 = int(self.alpha * y2 + (1 - self.alpha) * cy2)
        
        # Limit maximum movement per frame (prevent jumps)
        max_movement = 50  # pixels
        smooth_x1 = max(cx1 - max_movement, min(cx1 + max_movement, smooth_x1))
        smooth_y1 = max(cy1 - max_movement, min(cy1 + max_movement, smooth_y1))
        smooth_x2 = max(cx2 - max_movement, min(cx2 + max_movement, smooth_x2))
        smooth_y2 = max(cy2 - max_movement, min(cy2 + max_movement, smooth_y2))
        
        self.current_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
        return self.current_bbox

class TemporalMedianTracker:
    """Temporal median tracker for slide regions"""
    
    def __init__(self, window_size: int = 10, inertia_factor: float = 0.8):
        """
        Args:
            window_size: Number of frames to consider for median
            inertia_factor: Resistance to change (0 = no inertia, 1 = no change)
        """
        self.window_size = window_size
        self.inertia_factor = inertia_factor
        self.bbox_history: Deque[Tuple[int, int, int, int]] = deque(maxlen=window_size)
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
    
    def update(self, bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Update tracker with new detection
        
        Args:
            bbox: New bounding box or None
            
        Returns:
            Stabilized bounding box or None
        """
        if bbox is None:
            # No detection - return previous if available
            return self.current_bbox
        
        # Add to history
        self.bbox_history.append(bbox)
        
        if len(self.bbox_history) < 3:
            # Not enough history, return current
            self.current_bbox = bbox
            return bbox
        
        # Calculate median of recent detections
        x1_list = [b[0] for b in self.bbox_history]
        y1_list = [b[1] for b in self.bbox_history]
        x2_list = [b[2] for b in self.bbox_history]
        y2_list = [b[3] for b in self.bbox_history]
        
        median_bbox = (
            int(np.median(x1_list)),
            int(np.median(y1_list)),
            int(np.median(x2_list)),
            int(np.median(y2_list))
        )
        
        # Apply inertia if we have previous bbox
        if self.current_bbox is not None:
            cx1, cy1, cx2, cy2 = self.current_bbox
            mx1, my1, mx2, my2 = median_bbox
            
            # Weighted average between current and median
            stable_bbox = (
                int(self.inertia_factor * cx1 + (1 - self.inertia_factor) * mx1),
                int(self.inertia_factor * cy1 + (1 - self.inertia_factor) * my1),
                int(self.inertia_factor * cx2 + (1 - self.inertia_factor) * mx2),
                int(self.inertia_factor * cy2 + (1 - self.inertia_factor) * my2)
            )
            self.current_bbox = stable_bbox
            return stable_bbox
        else:
            self.current_bbox = median_bbox
            return median_bbox

# Test temporal smoothing
def test_temporal_smoothing():
    """Test temporal smoothing with synthetic data"""
    face_tracker = EMATracker(alpha=0.3)
    slide_tracker = TemporalMedianTracker(window_size=8)
    
    # Simulate noisy detections
    base_face = (100, 100, 200, 200)
    base_slide = (300, 50, 800, 400)
    
    print("Testing temporal smoothing...")
    
    for i in range(20):
        # Add noise to simulate real detections
        noise = np.random.randint(-10, 10, 4)
        noisy_face = tuple(np.array(base_face) + noise)
        noisy_slide = tuple(np.array(base_slide) + noise)
        
        smooth_face = face_tracker.update(noisy_face, 0.8)
        smooth_slide = slide_tracker.update(noisy_slide)
        
        print(f"Frame {i}: Face {noisy_face} -> {smooth_face}")
        print(f"Frame {i}: Slide {noisy_slide} -> {smooth_slide}")

if __name__ == "__main__":
    test_temporal_smoothing()