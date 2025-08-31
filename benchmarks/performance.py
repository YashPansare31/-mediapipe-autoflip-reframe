import time
import psutil
import sys
import os
sys.path.append('..')

from src.detectors.face_mp import FaceDetector
from src.detectors.zone_detector import UniversalWebinarDetector
from src.production_pipeline import ProductionPipeline

def benchmark_individual_components():
    """Benchmark individual components"""
    
    print("=== Component Performance Benchmark ===")
    
    # Test video
    test_video = "data/samples/s9.mp4"
    if not os.path.exists(test_video):
        print("Test video not found, creating synthetic test")
        return
    
    # Initialize components
    face_detector = FaceDetector(min_confidence=0.3)
    region_detector = UniversalWebinarDetector()
    
    import cv2
    cap = cv2.VideoCapture(test_video)
    
    # Benchmark settings
    test_frames = 100
    face_times = []
    region_times = []
    memory_usage = []
    
    print(f"Benchmarking {test_frames} frames...")
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Memory before
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Face detection timing
        start = time.time()
        face_bbox, face_conf = face_detector.detect(frame)
        face_time = time.time() - start
        face_times.append(face_time * 1000)  # Convert to ms
        
        # Region detection timing
        start = time.time()
        face_region, slide_region = region_detector.get_regions(frame, face_bbox)
        region_time = time.time() - start
        region_times.append(region_time * 1000)
        
        # Memory after
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage.append(mem_after)
        
        if i % 25 == 0:
            print(f"  Frame {i}: Face {face_time*1000:.1f}ms, Region {region_time*1000:.1f}ms")
    
    cap.release()
    face_detector.close()
    
    # Results
    print(f"\n=== Benchmark Results ===")
    print(f"Face Detection:")
    print(f"  Average: {sum(face_times)/len(face_times):.1f}ms")
    print(f"  Min: {min(face_times):.1f}ms")
    print(f"  Max: {max(face_times):.1f}ms")
    
    print(f"Region Detection:")
    print(f"  Average: {sum(region_times)/len(region_times):.1f}ms")
    print(f"  Min: {min(region_times):.1f}ms")
    print(f"  Max: {max(region_times):.1f}ms")
    
    print(f"Memory Usage:")
    print(f"  Average: {sum(memory_usage)/len(memory_usage):.1f}MB")
    print(f"  Peak: {max(memory_usage):.1f}MB")
    
    total_avg = sum(face_times)/len(face_times) + sum(region_times)/len(region_times)
    theoretical_fps = 1000 / total_avg
    print(f"Theoretical Max FPS: {theoretical_fps:.1f}")

def benchmark_full_pipeline():
    """Benchmark complete pipeline"""
    
    print("\n=== Full Pipeline Benchmark ===")
    
    config = {
        'min_face_conf': 0.3,
        'shot_threshold': 0.3,
        'face_alpha': 0.3,
        'slide_window': 8,
        'target_bitrate': '6M'
    }
    
    pipeline = ProductionPipeline(config)
    
    # Test with short clip
    test_input = "data/samples/s9.mp4"
    test_output = "Output/benchmark_output.mp4"
    
    if not os.path.exists(test_input):
        print("Test video not found")
        return
    
    start_time = time.time()
    
    success = pipeline.process_video_production(test_input, test_output)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if success and os.path.exists(test_output):
        input_size = os.path.getsize(test_input) / (1024 * 1024)
        output_size = os.path.getsize(test_output) / (1024 * 1024)
        
        print(f"Processing Time: {processing_time:.1f}s")
        print(f"Input Size: {input_size:.1f}MB")
        print(f"Output Size: {output_size:.1f}MB")
        print(f"Processing Speed: {input_size/processing_time:.2f} MB/s")
        
        # Clean up
        os.remove(test_output)
    else:
        print("Pipeline benchmark failed")

if __name__ == "__main__":
    benchmark_individual_components()
    benchmark_full_pipeline()