# Autopilot with YOLO and OpenCV (for linux)

This project was developed for educational purposes, utilizing the YOLO model, OpenCV, and PyTorch to detect and track traffic elements for use in autonomous driving simulation. 

### Project Structure
```bash 
├── main.py             # Main script
├── requirements.txt    # Required packages
└── README.md           # Project description
└── yolov8n.pt ()       # models (After running the program, you will see)
└── autopilot-env ()    # Virtual Environment Path (After create venv, you will see)
└── example             # Include your sample videos in this path
```

### Features
1. Object detection with using [YOLOv8](https://docs.ultralytics.com/)
2. Lane following woth using [OpenCV](https://opencv.org/)
3. Traffic detection

### Installation
1. Clone this repository:
```bash
git clone https://github.com/MojiMojisocute/autopilot.git && cd autopilot
```
2. Create [virtual environment](https://docs.python.org/3/library/venv.html): <br>
(Linux/macOS:) 
```bash
python3 -m venv autopilot-env
source autopilot-env/bin/activate
```

3. Install all required Python packages:
```bash
pip install -r requirements.txt
```

### How to use Program
Single video file testing:
```bash
# Test with video file
python autopilot_car.py --input "./example/yourvideoname.mp4" --output "output_result.mp4"

# Use large model for highest accuracy
python autopilot_car.py --input "./example/yourvideoname.mp4" --output "result.mp4" --model yolo_nas_l

# Test without saving results (view only)
python autopilot_car.py --input "./example/yourvideoname.mp4"
```

### Shortcut keys for testing.

```bash
' q ' : Exit the program

' p ' : Play/Pause video

' s ' : Capture screenshot (for camera)
```

### Test output data
#### While running :
1. FPS (Processing speed)
2. Progress bar (Video progress)
3. Real-time detection (Real-time detection)
4. Driving decisions (Driving decisions)
#### After finishing :
```bash
Processing Statistics:
Frames processed: 5079
Average processing time: 0.011s
Average FPS: 91.4
Min processing time: 0.009s
Max processing time: 2.357s
Output video saved to: output_result.mp4
Improved autopilot system stopped
```
### Recommended videos for testing
1. Appropriate lighting
2. High-quality video

### bug
1. Incorrect lane estimation
2. Traffic light detection is not accurate enough
3. The detection of car turn signals is not accurate enough because if the car's color matches the turn signal light, it causes detection errors.

### Example image
Video courtesy of [J Utah](https://www.youtube.com/watch?v=7HaJArMDKgI&t=1629s) and [7ze3 Travels](https://www.youtube.com/watch?v=b-WViLMs_4c)

#### Standard
<div align="center">

<img src="./image/Screenshot%20from%202025-07-17%2003-59-52.png" width="400" />
<img src="./image/Screenshot%20from%202025-07-17%2004-01-38.png" width="400" />

</div>

#### have bug (Traffic light detection is not working, turn signal detection is inaccurate, and lane detection is incorrect.)
<div align="center">

<img src="./image/Screenshot%20from%202025-07-17%2004-03-09.png" width="400" />

</div>

## License
[MIT](https://choosealicense.com/licenses/mit/)