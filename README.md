git # Video Frame Stitching for CCTV Camera Movement

This project extracts frames from a CCTV video only when the camera is moving, stitches them into a panorama when the camera stops, and saves each stitched image.

## Features
- Detects camera movement using frame-to-frame difference (motion detection)
- Extracts and collects frames only during camera movement
- Stitches frames into a panorama when the camera stops
- Saves each panorama to the `StitchedFrame` directory
- Handles multiple movement events in a single video (creates multiple panoramas)

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
- `--max-frames`: Maximum number of frames to process per movement (default: 100)
- `--output-dir`: Output directory for stitched images (default: `StitchedFrame`)

### Example
```
python frame_processor.py /path/to/video.mp4 --max-frames 100 --output-dir StitchedFrame
```

The stitched images will be saved in the `StitchedFrame` directory as `stitched_panorama_1.jpg`, `stitched_panorama_2.jpg`, etc.

## Output
- Each panorama image is saved in the `StitchedFrame` folder, one for each camera movement event.

## Notes
- The script uses motion detection to determine when the camera is moving.
- The quality of stitching depends on the video and camera movement.
- You can adjust the motion detection threshold in the `MotionDetector` class in `frame_processor.py` if needed.

## Troubleshooting
- If stitching fails, check the log for errors. Not enough overlap or too few frames can cause failure.
- For best results, use videos with smooth camera movement and good lighting.