# Video Frame Stitching for CCTV Camera Movement

This project extracts frames from a CCTV video, preprocesses them, stitches them into a panorama when the camera moves, and saves the stitched image.

## Features
- Extracts frames at intervals from a video file
- Preprocesses frames (resize, grayscale, denoise, enhance)
- Stitches frames into a panorama using OpenCV
- Saves the stitched image to a specified directory

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- numpy

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the frame processor script:
```
python frame_processor.py <video_path>
```

### Optional Arguments
- `--interval`: Frame extraction interval (default: 30)
- `--max-frames`: Maximum number of frames to process (default: 100)
- `--output`: Output path for the stitched image (default: `/home/affi/Desktop/VideoStitching/StitchedFrame/stitched_output.jpg`)

### Example
```
python frame_processor.py /path/to/video.mp4 --interval 20 --max-frames 50 --output /home/affi/Desktop/VideoStitching/StitchedFrame/stitched_output.jpg
```

The stitched image will be saved in the `StitchedFrame` directory.

## Output
- The stitched panorama image is saved as `stitched_output.jpg` in the `StitchedFrame` folder by default.

## Notes
- Ensure the `StitchedFrame` directory exists before running the script, or the script will attempt to create it.
- The quality of stitching depends on the video and camera movement.

## Troubleshooting
- If stitching fails, check the log for errors. Not enough overlap or too few frames can cause failure.
- For best results, use videos with smooth camera movement and good lighting.