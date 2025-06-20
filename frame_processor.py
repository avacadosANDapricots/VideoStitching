import cv2
import numpy as np
from typing import List, Optional, Tuple, Generator
import logging
from collections import deque
import argparse
import sys
import os

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

class MotionDetector:
    def __init__(self, threshold: float = 30.0, min_motion_frames: int = 3):
        """
        Detects motion based on frame-to-frame difference.
        Args:
            threshold (float): Threshold for mean absolute difference to detect motion.
            min_motion_frames (int): Minimum consecutive frames to confirm motion.
        """
        self.threshold = threshold
        self.min_motion_frames = min_motion_frames
        self.motion_counter = 0
        self.prev_frame = None
        self.in_motion = False

    def detect(self, frame: np.ndarray) -> bool:
        if self.prev_frame is None:
            self.prev_frame = frame
            return False
        diff = cv2.absdiff(self.prev_frame, frame)
        mean_diff = np.mean(diff)
        self.prev_frame = frame
        if mean_diff > self.threshold:
            self.motion_counter += 1
            if self.motion_counter >= self.min_motion_frames:
                self.in_motion = True
        else:
            if self.in_motion and self.motion_counter >= self.min_motion_frames:
                self.in_motion = False
                self.motion_counter = 0
                return 'motion_stopped'
            self.motion_counter = 0
        return self.in_motion


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

def process_video_motion_based(video_path: str, max_frames: int = 100, output_dir: str = "StitchedFrame"):
    """
    Process video and stitch frames only when camera is moving.
    Args:
        video_path (str): Path to the video file
        max_frames (int): Maximum number of frames to process per movement
        output_dir (str): Directory to save stitched images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_manager = FrameManager(max_frames)
    preprocessor = FramePreprocessor()
    motion_detector = MotionDetector()
    with FrameExtractor(video_path, interval=1) as extractor:
        movement_count = 0
        for idx, frame in enumerate(extractor.extract_frames()):
            processed_frame = preprocessor.preprocess_frame(frame)
            motion_status = motion_detector.detect(processed_frame)
            if motion_status is True:
                frame_manager.add_frame(frame)  # Use original color frame for stitching
            elif motion_status == 'motion_stopped' and len(frame_manager.frames) > 1:
                movement_count += 1
                stitcher = cv2.Stitcher_create()
                status, stitched = stitcher.stitch(frame_manager.get_frames())
                if status == cv2.Stitcher_OK:
                    out_path = os.path.join(output_dir, f"stitched_panorama_{movement_count}.jpg")
                    cv2.imwrite(out_path, stitched)
                    logger.info(f"Stitched image saved to {out_path}")
                else:
                    logger.error(f"Stitching failed for movement {movement_count} with status {status}")
                frame_manager.clear()
        # Handle any remaining frames at the end
        if len(frame_manager.frames) > 1:
            movement_count += 1
            stitcher = cv2.Stitcher_create()
            status, stitched = stitcher.stitch(frame_manager.get_frames())
            if status == cv2.Stitcher_OK:
                out_path = os.path.join(output_dir, f"stitched_panorama_{movement_count}.jpg")
                cv2.imwrite(out_path, stitched)
                logger.info(f"Stitched image saved to {out_path}")
            else:
                logger.error(f"Stitching failed for movement {movement_count} with status {status}")
            frame_manager.clear()

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description='Process video frames for panorama stitching based on camera movement')
        parser.add_argument('video_path', help='Path to the input video file')
        parser.add_argument('--interval', type=int, default=30,
                          help='Frame extraction interval (default: 30)')
        parser.add_argument('--max-frames', type=int, default=100,
                          help='Maximum number of frames to process per movement (default: 100)')
        parser.add_argument('--output-dir', type=str, default='StitchedFrame',
                          help='Output directory for stitched images')
        return parser.parse_args()
    try:
        args = parse_args()
        process_video_motion_based(
            args.video_path,
            max_frames=args.max_frames,
            output_dir=args.output_dir
        )
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        sys.exit(1)