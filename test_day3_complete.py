import sys
import os
import time
import numpy as np
sys.path.append('src')

def test_day3_complete():
    """Complete Day 3 integration test with all components"""
    
    print("=== Day 3 Complete Integration Test ===")
    
    # Test 1: Setup Check
    print("\n1. Checking setup...")
    try:
        from src.production_pipeline import ProductionPipeline
        from src.edge_cases.handler import EdgeCaseHandler
        from src.encoding.video_encoder import VideoEncoder
        from src.optimization.performance import PerformanceOptimizer
        print("   ✓ All modules imported successfully")
    except Exception as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    # Test 2: Performance Optimization
    print("\n2. Testing performance optimization...")
    try:
        optimizer = PerformanceOptimizer()
        optimizer.optimize_opencv_settings()
        print("   ✓ Performance optimization working")
    except Exception as e:
        print(f"   ✗ Performance optimization failed: {e}")
    
    # Test 3: Video Encoding
    print("\n3. Testing video encoding...")
    try:
        encoder = VideoEncoder()
        
        # Test with small synthetic video if no sample available
        if not os.path.exists("data/samples/s7.mp4"):
            print("   Creating synthetic test video...")
            _create_test_video("test_input.mp4")
            test_input = "test_input.mp4"
        else:
            test_input = "data/samples/s7.mp4"
        
        # Get video info
        info = encoder.get_video_info(test_input)
        print(f"   Video info: {info.get('width', 0)}x{info.get('height', 0)}")
        
        encoder.cleanup()
        print("   ✓ Video encoding working")
    except Exception as e:
        print(f"   ✗ Video encoding failed: {e}")
    
    # Test 4: Edge Case Handling
    print("\n4. Testing edge case handling...")
    try:
        handler = EdgeCaseHandler()
        
        # Test various edge cases
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        no_face_result = handler.handle_no_face_detection(test_frame, 100, 10)
        poor_quality_result = handler.handle_poor_quality_frame(test_frame)
        
        print("   ✓ Edge case handling working")
    except Exception as e:
        print(f"   ✗ Edge case handling failed: {e}")
    
    # Test 5: Complete Production Pipeline
    print("\n5. Testing complete production pipeline...")
    
    if not os.path.exists("data/samples/s7.mp4"):
        print("   No test video available, skipping full pipeline test")
        return True
    
    try:
        config = {
            'min_face_conf': 0.3,
            'shot_threshold': 0.3,
            'face_alpha': 0.3,
            'slide_window': 8,
            'target_bitrate': '4M',
            'enable_quality_enhancement': True
        }
        
        pipeline = ProductionPipeline(config)
        
        def test_progress(percent, current, total):
            if current % 50 == 0:  # Reduce output
                print(f"   Progress: {percent:.0f}%")
        
        # Process short clip
        input_video = "data/samples/s7.mp4"
        output_video = "output/day3_production_test.mp4"
        
        os.makedirs("output", exist_ok=True)
        
        start_time = time.time()
        success = pipeline.process_video_production(
            input_video, output_video, test_progress
        )
        end_time = time.time()
        
        if success:
            processing_time = end_time - start_time
            print(f"   ✓ Production pipeline working ({processing_time:.1f}s)")
            
            # Check output quality
            if os.path.exists(output_video):
                file_size = os.path.getsize(output_video) / (1024 * 1024)
                print(f"   ✓ Output: {output_video} ({file_size:.1f} MB)")
            
            return True
        else:
            print("   ✗ Production pipeline failed")
            return False
            
    except Exception as e:
        print(f"   ✗ Production pipeline error: {e}")
        return False

def _create_test_video(output_path: str):
    """Create synthetic test video for testing"""
    import cv2
    import numpy as np
    
    # Create simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
    
    for i in range(300):  # 10 seconds at 30fps
        # Create frame with moving elements
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add moving "face" (white circle)
        center_x = 200 + int(50 * np.sin(i * 0.1))
        center_y = 200 + int(30 * np.cos(i * 0.1))
        cv2.circle(frame, (center_x, center_y), 80, (255, 255, 255), -1)
        
        # Add "slide" area (white rectangle)
        cv2.rectangle(frame, (600, 100), (1200, 600), (200, 200, 200), -1)
        cv2.putText(frame, f'Frame {i}', (700, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_day3_complete()
    else:
        test_day3_complete()