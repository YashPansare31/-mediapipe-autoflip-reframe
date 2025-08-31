import cv2
import numpy as np
from typing import Tuple, Optional
import scipy.ndimage

# Target layout constants
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
FACE_HEIGHT = 640  # Top 1/3
SLIDE_HEIGHT = 1280  # Bottom 2/3

class EnhancedLayoutComposer:
    """Enhanced layout composer with better quality and stability"""
    
    def __init__(self):
        self.face_region = (0, 0, TARGET_WIDTH, FACE_HEIGHT)
        self.slide_region = (0, FACE_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT)
        
        # Quality enhancement settings
        self.enable_sharpening = True
        self.enable_noise_reduction = True
        self.enable_contrast_enhancement = True
    
    def compose_frame(self, source_frame: np.ndarray, 
                     face_bbox: Optional[Tuple[int, int, int, int]], 
                     slide_bbox: Optional[Tuple[int, int, int, int]],
                     enhance_quality: bool = True) -> np.ndarray:
        """
        Enhanced frame composition with quality improvements
        """
        # Create target canvas with gradient background
        output_frame = self._create_background()
        
        # Compose face region (top 1/3)
        if face_bbox is not None:
            face_crop = self._crop_and_resize_face(source_frame, face_bbox, enhance_quality)
            if face_crop is not None:
                self._place_face_region(output_frame, face_crop)
        
        # Compose slide region (bottom 2/3)  
        if slide_bbox is not None:
            slide_crop = self._crop_and_resize_slide(source_frame, slide_bbox, enhance_quality)
            if slide_crop is not None:
                self._place_slide_region(output_frame, slide_crop)
        
        return output_frame
    
    def _create_background(self) -> np.ndarray:
        """Create attractive gradient background"""
        # Create vertical gradient from dark to darker
        background = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        
        for y in range(TARGET_HEIGHT):
            # Gradient from 25 to 15 (dark gray to darker)
            intensity = int(25 - (y / TARGET_HEIGHT) * 10)
            background[y, :] = [intensity, intensity, intensity]
        
        return background
    
    def _crop_and_resize_face(self, source_frame: np.ndarray, 
                             face_bbox: Tuple[int, int, int, int],
                             enhance: bool = True) -> Optional[np.ndarray]:
        """Enhanced face cropping and resizing"""
        x1, y1, x2, y2 = face_bbox
        src_h, src_w = source_frame.shape[:2]
        
        # Clamp and validate coordinates
        x1 = max(0, min(x1, src_w-1))
        y1 = max(0, min(y1, src_h-1))
        x2 = max(x1+1, min(x2, src_w))
        y2 = max(y1+1, min(y2, src_h))
        
        # Crop face region
        face_crop = source_frame[y1:y2, x1:x2].copy()
        
        if face_crop.size == 0:
            return None
        
        # Calculate target size maintaining aspect ratio
        crop_h, crop_w = face_crop.shape[:2]
        crop_aspect = crop_w / crop_h
        target_aspect = TARGET_WIDTH / FACE_HEIGHT
        
        if crop_aspect > target_aspect:
            # Wider face - fit width
            new_w = TARGET_WIDTH
            new_h = int(TARGET_WIDTH / crop_aspect)
            new_h = min(new_h, FACE_HEIGHT)  # Don't exceed face region
        else:
            # Taller face - fit height with some margin
            new_h = min(FACE_HEIGHT - 40, int(crop_h * TARGET_WIDTH / crop_w))
            new_w = int(new_h * crop_aspect)
        
        # Resize with high quality
        resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply enhancements
        if enhance:
            resized = self._enhance_face_quality(resized)
        
        return resized
    
    def _crop_and_resize_slide(self, source_frame: np.ndarray, 
                              slide_bbox: Tuple[int, int, int, int],
                              enhance: bool = True) -> Optional[np.ndarray]:
        """Enhanced slide cropping with text sharpening"""
        x1, y1, x2, y2 = slide_bbox
        src_h, src_w = source_frame.shape[:2]
        
        # Clamp coordinates
        x1 = max(0, min(x1, src_w-1))
        y1 = max(0, min(y1, src_h-1))
        x2 = max(x1+1, min(x2, src_w))
        y2 = max(y1+1, min(y2, src_h))
        
        # Crop slide region
        slide_crop = source_frame[y1:y2, x1:x2].copy()
        
        if slide_crop.size == 0:
            return None
        
        # Resize to fill slide area (prioritize text readability)
        resized = cv2.resize(slide_crop, (TARGET_WIDTH, SLIDE_HEIGHT), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # Apply slide-specific enhancements
        if enhance:
            resized = self._enhance_slide_quality(resized)
        
        return resized
    
    def _enhance_face_quality(self, face_crop: np.ndarray) -> np.ndarray:
        """Apply quality enhancements to face region"""
        enhanced = face_crop.copy()

        # Gamma correction for dark faces
        gray_face = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        if gray_face.mean() < 120:
            # Apply gamma correction to brighten
            gamma = 1.5  # Brighten
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
    
        return enhanced
    
    def _enhance_slide_quality(self, slide_crop: np.ndarray) -> np.ndarray:
        """Apply slide-specific quality enhancements"""
        enhanced = slide_crop.copy()
        
        # Text sharpening (more aggressive than face)
        if self.enable_sharpening:
            # Unsharp mask for text clarity
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Contrast enhancement for text readability
        if self.enable_contrast_enhancement:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        
        return enhanced
    
    def _place_face_region(self, output_frame: np.ndarray, face_crop: np.ndarray):
        """Place face crop in top region with smooth edges"""
        crop_h, crop_w = face_crop.shape[:2]
        
        # Center in face region
        start_x = (TARGET_WIDTH - crop_w) // 2
        start_y = (FACE_HEIGHT - crop_h) // 2
        
        end_x = start_x + crop_w
        end_y = start_y + crop_h
        
        # Add subtle border/shadow effect
        # Create soft shadow
        shadow_offset = 3
        shadow_region = output_frame[start_y+shadow_offset:end_y+shadow_offset, 
                                   start_x+shadow_offset:end_x+shadow_offset]
        if shadow_region.shape[:2] == face_crop.shape[:2]:
            shadow_region[:] = cv2.addWeighted(shadow_region, 0.7, 
                                             np.zeros_like(face_crop), 0.3, 0)
        
        # Place face
        output_frame[start_y:end_y, start_x:end_x] = face_crop
    
    def _place_slide_region(self, output_frame: np.ndarray, slide_crop: np.ndarray):
        """Place slide crop in bottom region"""
        start_y = FACE_HEIGHT
        end_y = TARGET_HEIGHT
        
        output_frame[start_y:end_y, 0:TARGET_WIDTH] = slide_crop