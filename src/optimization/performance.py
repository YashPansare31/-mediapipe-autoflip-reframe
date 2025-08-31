import time
import psutil
import threading
from typing import Dict, List, Optional, Callable, Tuple
import queue
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    fps_processing: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    frame_processing_time_ms: float = 0.0
    total_frames_processed: int = 0

class PerformanceOptimizer:
    """Handles performance optimization and monitoring"""
    
    def __init__(self, enable_multithreading: bool = True):
        self.enable_multithreading = enable_multithreading
        self.metrics = PerformanceMetrics()
        self.frame_times: List[float] = []
        
    def optimize_opencv_settings(self):
        """Optimize OpenCV for better performance"""
        import cv2
        
        # Use optimized OpenCV if available
        cv2.setUseOptimized(True)
        
        # Set number of threads
        cv2.setNumThreads(min(4, psutil.cpu_count()))
        
        print(f"OK OpenCV optimized: {cv2.useOptimized()}")
        print(f"OK OpenCV threads: {cv2.getNumThreads()}")
    
    def create_threaded_detector(self, detector_class, *args, **kwargs):
        """Create threaded version of detector for parallel processing"""
        if not self.enable_multithreading:
            return detector_class(*args, **kwargs)
        
        return ThreadedDetector(detector_class, *args, **kwargs)
    
    def monitor_performance(self, start_time: float, frame_count: int):
        """Monitor and update performance metrics"""
        current_time = time.time()
        elapsed = current_time - start_time
        
        # FPS calculation
        if elapsed > 0:
            self.metrics.fps_processing = frame_count / elapsed
        
        # Memory usage
        process = psutil.Process()
        self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # CPU usage
        self.metrics.cpu_usage_percent = process.cpu_percent()
        
        # Frame processing time
        if self.frame_times:
            self.metrics.frame_processing_time_ms = sum(self.frame_times[-10:]) / len(self.frame_times[-10:]) * 1000
        
        self.metrics.total_frames_processed = frame_count
    
    def log_performance(self, frame_number: int):
        """Log current performance metrics"""
        if frame_number % 100 == 0:  # Log every 100 frames
            print(f"Performance @ frame {frame_number}:")
            print(f"  Processing FPS: {self.metrics.fps_processing:.1f}")
            print(f"  Memory: {self.metrics.memory_usage_mb:.1f} MB")
            print(f"  CPU: {self.metrics.cpu_usage_percent:.1f}%")
            print(f"  Avg frame time: {self.metrics.frame_processing_time_ms:.1f} ms")
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest performance optimizations based on metrics"""
        suggestions = []
        
        if self.metrics.fps_processing < 10:
            suggestions.append("Consider reducing input resolution or frame rate")
            suggestions.append("Enable multithreading if disabled")
        
        if self.metrics.memory_usage_mb > 2000:  # > 2GB
            suggestions.append("Reduce batch size or enable frame-by-frame processing")
        
        if self.metrics.cpu_usage_percent > 90:
            suggestions.append("Reduce detection frequency or use simpler algorithms")
        
        if self.metrics.frame_processing_time_ms > 200:  # > 200ms per frame
            suggestions.append("Optimize detector parameters")
            suggestions.append("Consider GPU acceleration if available")
        
        return suggestions

class ThreadedDetector:
    """Wrapper for threaded detection processing"""
    
    def __init__(self, detector_class, *args, **kwargs):
        self.detector = detector_class(*args, **kwargs)
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.is_running = True
    
    def _worker(self):
        """Worker thread for detection processing"""
        while self.is_running:
            try:
                frame, frame_id = self.input_queue.get(timeout=1.0)
                result = self.detector.detect(frame)
                self.output_queue.put((frame_id, result))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection thread error: {e}")
    
    def detect_async(self, frame, frame_id: int):
        """Submit frame for asynchronous detection"""
        try:
            self.input_queue.put((frame, frame_id), block=False)
        except queue.Full:
            pass  # Skip frame if queue full
    
    def get_result(self) -> Optional[Tuple[int, any]]:
        """Get detection result if available"""
        try:
            return self.output_queue.get(block=False)
        except queue.Empty:
            return None
    
    def close(self):
        """Close threaded detector"""
        self.is_running = False
        if hasattr(self.detector, 'close'):
            self.detector.close()

# Test performance optimization
def test_performance_optimization():
    """Test performance optimization features"""
    optimizer = PerformanceOptimizer(enable_multithreading=True)
    
    # Test OpenCV optimization
    optimizer.optimize_opencv_settings()
    
    # Simulate processing
    start_time = time.time()
    
    for i in range(100):
        frame_start = time.time()
        
        # Simulate frame processing
        time.sleep(0.01)  # 10ms processing time
        
        frame_end = time.time()
        optimizer.frame_times.append(frame_end - frame_start)
        
        # Monitor performance
        optimizer.monitor_performance(start_time, i + 1)
        
        if i % 25 == 0:
            optimizer.log_performance(i)
    
    # Get optimization suggestions
    suggestions = optimizer.suggest_optimizations()
    if suggestions:
        print("\nOptimization suggestions:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")

if __name__ == "__main__":
    test_performance_optimization()