
# API Reference

## Core Classes
    FaceDetector : Multi-method face detection with fallbacks.

    class FaceDetector:
        def __init__(self, min_confidence: float = 0.6)
        def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]


    UniversalWebinarDetector : Universal region detection that works regardless of webinar layout.

    class UniversalWebinarDetector:
        def get_regions(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -


### WebinarReframingPipeline
Main pipeline class for video processing.

class WebinarReframingPipeline:
    def __init__(self, config: dict = None)
    def process_video(self, input_path: str, output_path: str, use_autoflip: bool = True) -> bool


Parameters:

    config: Configuration dictionary with processing parameters
    input_path: Path to input webinar video
    output_path: Path for output reframed video
    use_autoflip: Whether to use AutoFlip (if available) or manual composition

Returns:

    bool: True if processing successful, False otherwise

