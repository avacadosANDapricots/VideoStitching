import cv2
import numpy as np
from typing import List, Optional, Tuple, Generator
import logging
from collections import deque
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, video_path: str, interval: int = 30):
        """
        Initialize the frame extractor.
        
        Args:
            video_path (str): Path to the video file or URL
            interval (int): Number of frames to skip between extractions
        """
        self.video_path = video_path
        self.interval = interval
        self.cap = None
        
    def __enter__(self):
        """Context manager entry"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_path}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.cap is not None:
            self.cap.release()
            
    def extract_frames(self, max_frames: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from the video at specified intervals.
        Yields frames one at a time to reduce memory usage.
        
        Args:
            max_frames (Optional[int]): Maximum number of frames to extract
            
        Yields:
            np.ndarray: Extracted frame
        """
        if self.cap is None:
            raise RuntimeError("VideoCapture is not initialized. Use FrameExtractor as a context manager or call __enter__() before extracting frames.")
        count = 0
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if count % self.interval == 0:
                frame_count += 1
                logger.info(f"Extracted frame {frame_count}")
                yield frame
                
                if max_frames and frame_count >= max_frames:
                    break
                    
            count += 1

class FramePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize the frame preprocessor.
        
        Args:
            target_size (Tuple[int, int]): Target size for frame resizing (width, height)
        """
        self.target_size = target_size
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        # Resize frame
        frame = cv2.resize(frame, self.target_size)
        
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

class FrameManager:
    def __init__(self, max_frames: int = 100):
        """
        Initialize the frame manager.
        
        Args:
            max_frames (int): Maximum number of frames to store in memory
        """
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the manager. The deque will automatically
        remove the oldest frame if max_frames is reached.
        
        Args:
            frame (np.ndarray): Frame to add
        """
        self.frames.append(frame)
            
    def get_frames(self) -> List[np.ndarray]:
        """
        Get all stored frames.
        
        Returns:
            List[np.ndarray]: List of stored frames
        """
        return list(self.frames)
    
    def clear(self) -> None:
        """Clear all stored frames."""
        self.frames.clear()

def process_video(video_path: str, interval: int = 30, max_frames: int = 100) -> List[np.ndarray]:
    """
    Process a video file and return preprocessed frames.
    
    Args:
        video_path (str): Path to the video file
        interval (int): Frame extraction interval
        max_frames (int): Maximum number of frames to process
        
    Returns:
        List[np.ndarray]: List of preprocessed frames
    """
    frame_manager = FrameManager(max_frames)
    preprocessor = FramePreprocessor()
    
    with FrameExtractor(video_path, interval) as extractor:
        for frame in extractor.extract_frames(max_frames):
            processed_frame = preprocessor.preprocess_frame(frame)
            frame_manager.add_frame(processed_frame)
            
    return frame_manager.get_frames()

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description='Process video frames for panorama stitching')
        parser.add_argument('video_path', help='Path to the input video file')
        parser.add_argument('--interval', type=int, default=30,
                          help='Frame extraction interval (default: 30)')
        parser.add_argument('--max-frames', type=int, default=100,
                          help='Maximum number of frames to process (default: 100)')
        parser.add_argument('--output', type=str, default='stitched_output.jpg',
                          help='Output path for stitched image')
        return parser.parse_args()
    
    try:
        args = parse_args()
        processed_frames = process_video(
            args.video_path,
            interval=args.interval,
            max_frames=args.max_frames
        )
        logger.info(f"Successfully processed {len(processed_frames)} frames")

        # --- Stitching ---
        if len(processed_frames) > 1:
            stitcher = cv2.Stitcher_create()
            status, stitched = stitcher.stitch(processed_frames)
            if status == cv2.Stitcher_OK:
                cv2.imwrite(args.output, stitched)
                logger.info(f"Stitched image saved to {args.output}")
            else:
                logger.error(f"Stitching failed with status {status}")
        else:
            logger.error("Not enough frames to stitch.")

    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        sys.exit(1)