from frame_processor import process_video

# Process a video file
video_path = "your_video.mp4"
processed_frames = process_video(
    video_path,
    interval=30,  # Extract every 30th frame
    max_frames=1000  # Store maximum 1000 frames
)