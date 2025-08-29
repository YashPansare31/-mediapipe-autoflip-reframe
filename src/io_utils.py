import cv2
import numpy as np
from typing import Generator, Tuple, Optional

class VideoReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield (frame, frame_number) tuples"""
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame, frame_num
            frame_num += 1
    
    def close(self):
        self.cap.release()

class VideoWriter:
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def write_frame(self, frame: np.ndarray):
        self.writer.write(frame)
    
    def close(self):
        self.writer.release()

# Test function
def test_video_io():
    # Test with a sample video if available
    pass