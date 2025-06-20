from frame_processor import process_video_motion_based

# Process a video file and create panoramas based on camera movement
video_path = "your_video.mp4"  # Replace with your video file path
process_video_motion_based(
    video_path,
    max_frames=100,  # Maximum frames per movement event
    output_dir="StitchedFrame"  # Output directory for panoramas
)