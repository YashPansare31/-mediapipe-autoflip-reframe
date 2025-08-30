import cv2
import numpy as np
from typing import Tuple, Optional

# Target layout constants
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
FACE_HEIGHT = 640  # Top 1/3
SLIDE_HEIGHT = 1280  # Bottom 2/3

class LayoutComposer:
    """Handles mapping source ROIs to target 9:16 layout"""
    
    def __init__(self):
        self.face_region = (0, 0, TARGET_WIDTH, FACE_HEIGHT)
        self.slide_region = (0, FACE_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT)
    
    def compose_frame(self, source_frame: np.ndarray, 
                     face_bbox: Optional[Tuple[int, int, int, int]], 
                     slide_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Compose final 9:16 frame from source frame and ROIs
        
        Args:
            source_frame: Original webinar frame
            face_bbox: Speaker bounding box (x1, y1, x2, y2)
            slide_bbox: Slide bounding box (x1, y1, x2, y2)
            
        Returns:
            Composed 1080x1920 frame
        """
        # Create target canvas
        output_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        
        # Fill with default background
        output_frame[:] = (40, 40, 40)  # Dark gray background
        
        # Compose face region (top 1/3)
        if face_bbox is not None:
            face_crop = self._crop_and_resize_face(source_frame, face_bbox)
            if face_crop is not None:
                self._place_face_region(output_frame, face_crop)
        
        # Compose slide region (bottom 2/3)
        if slide_bbox is not None:
            slide_crop = self._crop_and_resize_slide(source_frame, slide_bbox)
            if slide_crop is not None:
                self._place_slide_region(output_frame, slide_crop)
        
        return output_frame
    
    def _crop_and_resize_face(self, source_frame: np.ndarray, 
                             face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop face region and resize to fit top area
        """
        x1, y1, x2, y2 = face_bbox
        src_h, src_w = source_frame.shape[:2]
        
        # Clamp coordinates
        x1 = max(0, min(x1, src_w))
        y1 = max(0, min(y1, src_h))
        x2 = max(x1, min(x2, src_w))
        y2 = max(y1, min(y2, src_h))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop face region
        face_crop = source_frame[y1:y2, x1:x2]
        
        # Calculate resize to maintain aspect ratio in top region
        crop_h, crop_w = face_crop.shape[:2]
        crop_aspect = crop_w / crop_h
        target_aspect = TARGET_WIDTH / FACE_HEIGHT
        
        if crop_aspect > target_aspect:
            # Face is wider - fit width, letterbox height
            new_w = TARGET_WIDTH
            new_h = int(TARGET_WIDTH / crop_aspect)
        else:
            # Face is taller - fit height, pillarbox width  
            new_h = FACE_HEIGHT
            new_w = int(FACE_HEIGHT * crop_aspect)
        
        # Resize
        resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def _place_face_region(self, output_frame: np.ndarray, face_crop: np.ndarray):
        """
        Place face crop in top region with centering
        """
        crop_h, crop_w = face_crop.shape[:2]
        
        # Center horizontally and vertically in face region
        start_x = (TARGET_WIDTH - crop_w) // 2
        start_y = (FACE_HEIGHT - crop_h) // 2
        
        # Place in output frame
        end_x = start_x + crop_w
        end_y = start_y + crop_h
        
        output_frame[start_y:end_y, start_x:end_x] = face_crop
    
    def _crop_and_resize_slide(self, source_frame: np.ndarray, 
                              slide_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop slide region and resize to fit bottom area
        """
        x1, y1, x2, y2 = slide_bbox
        src_h, src_w = source_frame.shape[:2]
        
        # Clamp coordinates
        x1 = max(0, min(x1, src_w))
        y1 = max(0, min(y1, src_h))
        x2 = max(x1, min(x2, src_w))
        y2 = max(y1, min(y2, src_h))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop slide region
        slide_crop = source_frame[y1:y2, x1:x2]
        
        # Resize to fill slide area (may stretch slightly for readability)
        resized = cv2.resize(slide_crop, (TARGET_WIDTH, SLIDE_HEIGHT), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def _place_slide_region(self, output_frame: np.ndarray, slide_crop: np.ndarray):
        """
        Place slide crop in bottom region
        """
        start_y = FACE_HEIGHT
        end_y = TARGET_HEIGHT
        
        output_frame[start_y:end_y, 0:TARGET_WIDTH] = slide_crop

# Test layout composition
def test_layout_composer():
    """Test layout composition with synthetic data"""
    composer = LayoutComposer()
    
    # Create mock source frame
    source = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Mock ROIs
    face_bbox = (800, 200, 1100, 500)  # Right side face
    slide_bbox = (50, 50, 700, 600)    # Left side slide
    
    # Compose
    result = composer.compose_frame(source, face_bbox, slide_bbox)
    
    print(f"Result shape: {result.shape}")
    print("Layout composition test completed")
    
    # Save test result
    cv2.imwrite('layout_test.jpg', result)
    print("Saved layout_test.jpg")

if __name__ == "__main__":
    test_layout_composer()