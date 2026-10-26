# AI-Based Online Interview Behavioral Analysis & Suspicious Activity Detection

A web-based system that analyzes a candidate's behavior during an online interview using computer vision. It shows live webcam preview, tracks gaze and behavior signals, and reports possible suspicious activity on a dashboard.

> The results are **signals, not proof**. Looking away can have normal reasons, such as thinking or reading a question. A human should make the final decision.

## Features

- **Face and gaze analysis:** face detection and Left / Center / Right gaze estimation, logged to CSV
- **Behavior detection:** face absent, multiple faces, head turned, looking up or down, and blink rate
- **Suspicious activity detection:** object detection with YOLO and a combined score (in progress)
- **Dashboard:** candidate form, webcam preview, start/stop controls, timer, and three result cards

## Tech Stack

| Part | Tools |
|---|---|
| Frontend | HTML, CSS, JavaScript, WebRTC |
| Backend | Python, Flask |
| Computer vision | OpenCV, MediaPipe Face Landmarker, YOLO |
| Optional ML | scikit-learn |

## Project Structure

```
AI-INTERVIEW-ANALYSIS/
├── frontend/                  Dashboard (index.html, style.css, script.js, config.js)
├── backend/                   Flask API (app.py)
├── modules/
│   ├── gaze/                  Face and gaze analysis
│   └── behavior/              Behavior detection
├── models/                    face_landmarker.task
├── data/                      Generated CSV logs
├── archive/                   Old backup files
├── requirements.txt
└── README.md
```

## Team and Modules

| Member | Module |
|---|---|
| Vaishnavi | Face and gaze analysis |
| Sanjana | Behavior detection |
| Tripti | Backend and YOLO |
| Riya | Frontend and Git coordination |

## Setup

Use Python 3.9 to 3.12.

```bash
git clone https://github.com/riyajoshi07/AI-INTERVIEW-ANALYSIS.git
cd AI-INTERVIEW-ANALYSIS

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Download the MediaPipe, FER+ expression CNN, and YOLO model weights into `models/`:

```bash
python -m backend.download_models
```

The downloader refreshes the tracked MediaPipe model and stores the FER+ and YOLO weights locally; those larger model files are ignored by Git and need to be downloaded on each machine.

## How to Run

Run all commands from the project root folder.

**Behavior detection (webcam test)**
```bash
python -m modules.behavior.behavior_detection
```
Look at the camera for 2 seconds (calibration). Press `q` to quit.

**Gaze detection (webcam test)**
```bash
python -m modules.gaze.gaze_detection
```
Results are saved in `data/gaze_results.csv`. Press `q` to quit.

## Backend API

Run the API and dashboard from one terminal at the project root after installing `requirements.txt`:

```bash
python -m backend.app
```

Open `http://127.0.0.1:5000/` and allow camera permission in the browser. Flask serves the dashboard and API together. If port 5000 is already in use, set `PORT=5001` before running the command and open `http://127.0.0.1:5001/`. The frontend sends a resized camera frame once per second while an interview is active. All three model files are required; `/api/health` lists missing models and dependencies. Close other programs using the webcam before testing.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Check that the backend is running |
| POST | `/api/interview/start` | Start a session |
| POST | `/api/analysis/frame` | Analyze a JPEG frame (`frame` and `session_id` form fields) |
| POST | `/api/interview/stop` | Stop a session |
| GET | `/api/analysis/latest` | Get the latest results |

Expected response from `/api/analysis/latest`:

```json
{
  "gaze": { "value": "Center", "status": "Looking at screen" },
  "expression": { "value": "Neutral", "confidence": 0.8 },
  "behavior": { "value": "Normal Posture", "status": "Behavior looks normal" },
  "detected_objects": [],
  "suspicious_activity": { "value": "Low", "score": 0, "status": "No sustained signal detected" }
}
```

Results are written to `data/session_results.csv`. The combined score is a transparent heuristic based on face presence, sustained head movement, and phone/book detections. Expression output is not included in the score. These outputs are review signals, not proof of misconduct.

## Current Status

- [x] Frontend dashboard
- [x] Face detection and gaze estimation
- [x] Behavior detection module
- [x] Flask backend and dashboard integration
- [x] Facial-expression CNN inference
- [x] YOLO phone/book object detection
- [x] Combined suspicious activity score
- [x] Integration of all modules
- [ ] Demo video and final report

## Limitations

- Gaze and head pose are approximate estimates from face landmarks.
- Lighting, camera angle, and glasses can affect accuracy.
- The system should be used only with the candidate's consent and a clear privacy policy.

## Future Work

- More accurate head pose with `solvePnP`
- Validate phone and book detections under different camera conditions
- Session report with charts
- Fairness and accuracy testing on different users
