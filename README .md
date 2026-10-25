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

Download the MediaPipe model into the `models/` folder:

```bash
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

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

**Frontend**
1. Open the `frontend/` folder in VS Code.
2. Install the Live Server extension.
3. Right-click `index.html` and choose Open with Live Server.
4. Allow camera permission in the browser.

Only one program can use the webcam at a time. Close the others before testing.

## Backend API (planned)

The frontend is already built for these routes. In `frontend/config.js`, set `ENABLE_BACKEND: true` when the backend is ready.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Check that the backend is running |
| POST | `/api/interview/start` | Start a session |
| POST | `/api/interview/stop` | Stop a session |
| GET | `/api/analysis/latest` | Get the latest results |

Expected response from `/api/analysis/latest`:

```json
{
  "gaze": { "value": "Center", "status": "Looking at screen" },
  "behavior": { "value": "Normal Posture", "status": "Behavior looks normal" },
  "suspicious_activity": { "value": "Low", "status": "No issue found" }
}
```

## Current Status

- [x] Frontend dashboard
- [x] Face detection and gaze estimation
- [x] Behavior detection module
- [ ] Flask backend
- [ ] YOLO object detection
- [ ] Combined suspicious activity score
- [ ] Integration of all modules
- [ ] Demo video and final report

## Limitations

- Gaze and head pose are approximate estimates from face landmarks.
- Lighting, camera angle, and glasses can affect accuracy.
- The system should be used only with the candidate's consent and a clear privacy policy.

## Future Work

- More accurate head pose with `solvePnP`
- Phone and book detection with YOLO
- Session report with charts
- Fairness and accuracy testing on different users
