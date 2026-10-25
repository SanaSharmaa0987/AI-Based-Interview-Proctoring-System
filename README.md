
# AI-Based Online Interview Behavioral Analysis & Suspicious Activity Detection

## Project Overview
This project aims to analyze candidate behavior during an online interview using computer vision. It detects the face and estimates the candidate's gaze direction through a webcam.

## My Module: Face Detection and Gaze Estimation

### Features
- Real-time face detection
- Approximate gaze direction estimation
- Gaze classification: Left, Center, and Right
- Face presence detection
- CSV file logging with timestamps

### Technologies Used
- Python
- OpenCV
- MediaPipe Face Landmarker
- CSV Module

## Files
- `face_gaze.py` - Basic face detection
- `gaze_detection.py` - Face detection and gaze estimation
- `gaze_logging.py` - CSV logging test
- `gaze_results.csv` - Recorded gaze results
- `face_landmarker.task` - MediaPipe face landmark model

## How to Run

1. Install the required packages.
2. Ensure `face_landmarker.task` is in the project folder.
3. Run the following command:

```bash
python gaze_detection.py
```

4. Allow webcam access when prompted.
5. Press `q` to close the webcam window.

## Output
The system displays the detected face and estimated gaze direction. Results can be recorded in a CSV file with timestamps.

## Note
Gaze direction is an approximate computer-vision estimate and does not independently establish suspicious behavior.
