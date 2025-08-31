import sys
import numpy as np
from typing import Optional, Tuple, List, Deque, Dict
from collections import deque
import cv2

class AdvancedEMATracker:
    """Enhanced EMA tracker with adaptive smoothing"""
    
    def __init__(self, alpha: float = 0.25, adaptive: bool = True):
        self.base_alpha = alpha
        self.adaptive = adaptive
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.confidence_history: Deque[float] = deque(maxlen=12)
        self.velocity_history: Deque[float] = deque(maxlen=6)
        self.lost_frames = 0
    
    def update(self, bbox: Optional[Tuple[int, int, int, int]], confidence: float = 1.0) -> Optional[Tuple[int, int, int, int]]:
        """Enhanced update with adaptive smoothing"""
        self.confidence_history.append(confidence)
        
        if bbox is None:
            self.lost_frames += 1
            # Keep previous detection for short periods
            if self.lost_frames < 15 and self.current_bbox is not None:
                return self.current_bbox
            else:
                self.current_bbox = None
                return None
        
        self.lost_frames = 0
        
        if self.current_bbox is None:
            self.current_bbox = bbox
            return bbox
        
        # Calculate movement velocity for adaptive smoothing
        current_center = self._get_center(bbox)
        prev_center = self._get_center(self.current_bbox)
        velocity = np.linalg.norm(np.array(current_center) - np.array(prev_center))
        self.velocity_history.append(velocity)
        
        # Adaptive alpha based on movement and confidence
        alpha = self.base_alpha
        if self.adaptive:
            avg_velocity = sum(self.velocity_history) / len(self.velocity_history)
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            
            # Higher alpha for fast movement and high confidence
            if avg_velocity > 50 and avg_confidence > 0.7:
                alpha = min(0.7, self.base_alpha * 2)
            # Lower alpha for slow movement (more smoothing)
            elif avg_velocity < 10:
                alpha = max(0.1, self.base_alpha * 0.5)
        
        # Apply smoothing
        smoothed = self._apply_ema_smoothing(bbox, alpha)
        
        # Movement limiting
        smoothed = self._limit_movement(smoothed, max_movement=40)
        
        self.current_bbox = smoothed
        return smoothed
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _apply_ema_smoothing(self, bbox: Tuple[int, int, int, int], alpha: float) -> Tuple[int, int, int, int]:
        """Apply exponential moving average smoothing"""
        x1, y1, x2, y2 = bbox
        cx1, cy1, cx2, cy2 = self.current_bbox
        
        smooth_x1 = int(alpha * x1 + (1 - alpha) * cx1)
        smooth_y1 = int(alpha * y1 + (1 - alpha) * cy1)
        smooth_x2 = int(alpha * x2 + (1 - alpha) * cx2)
        smooth_y2 = int(alpha * y2 + (1 - alpha) * cy2)
        
        return (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
    
    def _limit_movement(self, bbox: Tuple[int, int, int, int], max_movement: int) -> Tuple[int, int, int, int]:
        """Limit maximum movement per frame"""
        if self.current_bbox is None:
            return bbox
        
        x1, y1, x2, y2 = bbox
        cx1, cy1, cx2, cy2 = self.current_bbox
        
        max_movement = 30

        # Limit each coordinate
        limited_x1 = max(cx1 - max_movement, min(cx1 + max_movement, x1))
        limited_y1 = max(cy1 - max_movement, min(cy1 + max_movement, y1))
        limited_x2 = max(cx2 - max_movement, min(cx2 + max_movement, x2))
        limited_y2 = max(cy2 - max_movement, min(cy2 + max_movement, y2))
        
        return (limited_x1, limited_y1, limited_x2, limited_y2)

class StabilityAnalyzer:
    """Analyzes and improves detection stability"""
    
    def __init__(self, stability_window: int = 30):
        self.stability_window = stability_window
        self.bbox_history: Deque = deque(maxlen=stability_window)
        self.jitter_threshold = 20  # pixels
    
    def analyze_stability(self, bbox: Optional[Tuple[int, int, int, int]]) -> Dict:
        """Analyze detection stability metrics"""
        if bbox is None:
            return {"stable": False, "jitter": 0, "confidence": 0}
        
        self.bbox_history.append(bbox)
        
        if len(self.bbox_history) < 5:
            return {"stable": True, "jitter": 0, "confidence": 0.5}
        
        # Calculate jitter (movement between frames)
        movements = []
        for i in range(1, len(self.bbox_history)):
            prev_center = self._get_center(self.bbox_history[i-1])
            curr_center = self._get_center(self.bbox_history[i])
            movement = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
            movements.append(movement)
        
        avg_jitter = sum(movements) / len(movements)
        max_jitter = max(movements)
        
        # Stability assessment
        is_stable = avg_jitter < self.jitter_threshold and max_jitter < self.jitter_threshold * 2
        
        return {
            "stable": is_stable,
            "jitter": avg_jitter,
            "max_jitter": max_jitter,
            "confidence": 1.0 if is_stable else max(0.1, 1.0 - (avg_jitter / 100))
        }
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class TemporalMedianTracker:
    """Temporal median tracker for slide regions"""
    
    def __init__(self, window_size: int = 12, inertia_factor: float = 0.85):
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
            return self.current_bbox
        
        self.bbox_history.append(bbox)
        
        if len(self.bbox_history) < 3:
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
        
        # Apply inertia
        if self.current_bbox is not None:
            cx1, cy1, cx2, cy2 = self.current_bbox
            mx1, my1, mx2, my2 = median_bbox
            
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


# Performance testing
def benchmark_detection_speed():
    """Benchmark detection performance"""
    import cv2
    import time
    sys.path.append('../..')
    
    from src.detectors.face_mp import FaceDetector
    from src.detectors.zone_detector import UniversalWebinarDetector
    
    # Initialize detectors
    face_detector = FaceDetector(min_confidence=0.3)
    region_detector = UniversalWebinarDetector()
    stability_analyzer = StabilityAnalyzer()
    
    # Test video
    cap = cv2.VideoCapture("data/samples/s5.mp4")
    if not cap.isOpened():
        print("Cannot open test video")
        return
    
    # Benchmark settings
    frames_to_test = 300
    face_times = []
    region_times = []
    total_times = []
    
    print(f"Benchmarking detection speed over {frames_to_test} frames...")
    
    overall_start = time.time()
    
    for frame_num in range(frames_to_test):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # Time face detection
        face_start = time.time()
        face_bbox, face_conf = face_detector.detect(frame)
        face_end = time.time()
        face_times.append(face_end - face_start)
        
        # Time region detection
        region_start = time.time()
        face_region, slide_region = region_detector.get_regions(frame, face_bbox)
        region_end = time.time()
        region_times.append(region_end - region_start)
        
        # Analyze stability
        stability = stability_analyzer.analyze_stability(face_region)
        
        frame_end = time.time()
        total_times.append(frame_end - frame_start)
        
        # Progress every 50 frames
        if frame_num % 50 == 0:
            avg_total = sum(total_times[-50:]) / len(total_times[-50:]) * 1000
            print(f"Frame {frame_num}: {avg_total:.1f}ms avg processing time")
    
    overall_end = time.time()
    
    cap.release()
    face_detector.close()
    
    # Calculate statistics
    avg_face_time = sum(face_times) / len(face_times) * 1000
    avg_region_time = sum(region_times) / len(region_times) * 1000
    avg_total_time = sum(total_times) / len(total_times) * 1000
    
    processing_fps = frames_to_test / (overall_end - overall_start)
    
    print(f"\n=== Performance Benchmark Results ===")
    print(f"Frames processed: {len(total_times)}")
    print(f"Face detection: {avg_face_time:.1f}ms avg")
    print(f"Region detection: {avg_region_time:.1f}ms avg") 
    print(f"Total per frame: {avg_total_time:.1f}ms avg")
    print(f"Processing FPS: {processing_fps:.1f}")
    print(f"Real-time capable: {'Yes' if processing_fps > 25 else 'No'}")

if __name__ == "__main__":
    benchmark_detection_speed()