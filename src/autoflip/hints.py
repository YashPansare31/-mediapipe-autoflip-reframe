import json
import os
from typing import List, Dict, Tuple, Optional

class AutoFlipHints:
    """Generates AutoFlip external hints from face/slide detections"""
    
    def __init__(self, output_dir: str = "temp"):
        self.output_dir = output_dir
        self.hints: List[Dict] = []
        os.makedirs(output_dir, exist_ok=True)
    
    def add_frame_detections(self, frame_number: int, timestamp: float,
                           face_bbox: Optional[Tuple[int, int, int, int]], face_confidence: float,
                           slide_bbox: Optional[Tuple[int, int, int, int]], 
                           frame_width: int, frame_height: int):
        """
        Add detections for a frame
        
        Args:
            frame_number: Frame index
            timestamp: Time in seconds
            face_bbox: Face bounding box (x1, y1, x2, y2) or None
            face_confidence: Face detection confidence
            slide_bbox: Slide bounding box or None
            frame_width: Source frame width
            frame_height: Source frame height
        """
        frame_hint = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "regions": []
        }
        
        # Add face region
        if face_bbox is not None and face_confidence > 0.3:
            x1, y1, x2, y2 = face_bbox
            # Normalize to [0, 1]
            norm_x1 = x1 / frame_width
            norm_y1 = y1 / frame_height
            norm_x2 = x2 / frame_width  
            norm_y2 = y2 / frame_height
            
            face_region = {
                "type": "FACE",
                "bbox": [norm_x1, norm_y1, norm_x2, norm_y2],
                "confidence": face_confidence,
                "priority": 100
            }
            frame_hint["regions"].append(face_region)
        
        # Add slide region
        if slide_bbox is not None:
            x1, y1, x2, y2 = slide_bbox
            # Normalize to [0, 1]
            norm_x1 = x1 / frame_width
            norm_y1 = y1 / frame_height  
            norm_x2 = x2 / frame_width
            norm_y2 = y2 / frame_height
            
            slide_region = {
                "type": "SCREEN",
                "bbox": [norm_x1, norm_y1, norm_x2, norm_y2],
                "confidence": 0.9,  # High confidence for slides
                "priority": 80
            }
            frame_hint["regions"].append(slide_region)
        
        self.hints.append(frame_hint)
    
    def save_hints(self, filename: str = "external_hints.json") -> str:
        """
        Save hints to JSON file for AutoFlip
        
        Returns:
            Path to saved hints file
        """
        hints_path = os.path.join(self.output_dir, filename)
        
        with open(hints_path, 'w') as f:
            json.dump({
                "external_hints": self.hints,
                "version": "1.0"
            }, f, indent=2)
        
        print(f"Saved {len(self.hints)} frame hints to {hints_path}")
        return hints_path
    
    def clear(self):
        """Clear accumulated hints"""
        self.hints.clear()

# Test hint generation
def test_hint_generation():
    """Test AutoFlip hint generation"""
    hints = AutoFlipHints()
    
    # Simulate some detections
    for i in range(10):
        face_bbox = (100 + i*5, 100, 200 + i*5, 250) if i % 2 == 0 else None
        slide_bbox = (300, 50, 800, 400)
        
        hints.add_frame_detections(
            frame_number=i,
            timestamp=i / 30.0,  # 30 FPS
            face_bbox=face_bbox,
            face_confidence=0.8,
            slide_bbox=slide_bbox,
            frame_width=1280,
            frame_height=720
        )
    
    hints_path = hints.save_hints()
    print(f"Generated test hints: {hints_path}")

if __name__ == "__main__":
    test_hint_generation()