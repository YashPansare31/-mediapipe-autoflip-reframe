import cv2
import numpy as np
import os
import sys
sys.path.append('..')

from src.production_pipeline import ProductionPipeline

def create_demo_video():
    """Create demonstration video showing before/after"""
    
    print("Creating demonstration video...")
    
    # Demo configuration
    config = {
        'min_face_conf': 0.3,
        'shot_threshold': 0.3,
        'face_alpha': 0.3,
        'slide_window': 8,
        'target_bitrate': '6M',
        'enable_quality_enhancement': True
    }
    
    input_video = "data/samples/s9.mp4"
    
    if not os.path.exists(input_video):
        print("Demo input video not found")
        return False
    
    # Create reframed version
    reframed_video = "Output/demo_reframed.mp4"
    pipeline = ProductionPipeline(config)
    
    success = pipeline.process_video_production(input_video, reframed_video)
    
    if not success:
        print("Failed to create reframed video")
        return False
    
    # Create side-by-side comparison
    comparison_video = "demo/demo_comparison.mp4"
    create_side_by_side_comparison(input_video, reframed_video, comparison_video)
    
    print(f"Demo videos created:")
    print(f"  Reframed: {reframed_video}")
    print(f"  Comparison: {comparison_video}")
    
    return True

def create_side_by_side_comparison(original: str, reframed: str, output: str):
    """Create side-by-side comparison video"""
    
    cap_orig = cv2.VideoCapture(original)
    cap_reframed = cv2.VideoCapture(reframed)
    
    # Get properties
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output dimensions (side by side)
    out_w = orig_w + 540  # Original width + half of reframed width
    out_h = max(orig_h, 1920)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (out_w, out_h))
    
    frame_count = 0
    
    while True:
        ret1, frame_orig = cap_orig.read()
        ret2, frame_reframed = cap_reframed.read()
        
        if not ret1 or not ret2:
            break
        
        # Create comparison frame
        comparison_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # Place original (left side)
        y_offset_orig = (out_h - orig_h) // 2
        comparison_frame[y_offset_orig:y_offset_orig+orig_h, 0:orig_w] = frame_orig
        
        # Resize and place reframed (right side)
        reframed_resized = cv2.resize(frame_reframed, (540, 960))  # Half size
        y_offset_reframed = (out_h - 960) // 2
        comparison_frame[y_offset_reframed:y_offset_reframed+960, orig_w:orig_w+540] = reframed_resized
        
        # Add labels
        cv2.putText(comparison_frame, 'Original (16:9)', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(comparison_frame, 'Reframed (9:16)', (orig_w + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        out.write(comparison_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Comparison frame {frame_count}")
    
    cap_orig.release()
    cap_reframed.release()
    out.release()

if __name__ == "__main__":
    create_demo_video()