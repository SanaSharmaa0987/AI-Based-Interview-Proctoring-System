# AI-Based Online Interview Behavioral Analysis & Suspicious Activity Detection

A web-based system that analyzes a candidate's behavior during an online interview using computer vision. It shows a live webcam preview, tracks gaze, behavior, expression and object signals, scores suspicious activity, and shows a warning or a black screen when suspicious activity continues for too long.

> The results are **signals, not proof**. Looking away can have normal reasons, such as thinking or reading a question. A human should make the final decision.

## Features

- **Face and gaze analysis:** face detection and Left / Center / Right gaze estimation, logged to CSV
- **Behavior detection:** face absent, multiple faces, head turned, looking up or down, and blink rate
- **Facial expression:** FER+ CNN estimate of the visible face (not used in the risk score)
- **Object detection:** YOLO detection of phones and books in the camera view
- **Suspicious activity score:** a transparent score (Low, Elevated, High) built from face presence, sustained head movement and detected objects
- **Warning and blackout:** a warning banner and a black screen when suspicious activity continues without a break (see the section below)
- **Dashboard:** candidate form, webcam preview, start and stop controls, timer, and result cards
- **Session log:** every analyzed frame is saved to `data/session_results.csv`

## Tech Stack

| Part | Tools |
|---|---|
| Frontend | HTML, CSS, JavaScript, WebRTC |
| Backend | Python, Flask |
| Computer vision | OpenCV, MediaPipe Face Landmarker, FER+ (ONNX), YOLOv8 (Ultralytics) |
| Optional ML | scikit-learn |

## Project Structure

```
AI-Based-Interview-Proctoring-System/
|-- frontend/                  Dashboard (index.html, style.css, script.js, config.js)
|-- backend/
|   |-- app.py                 Flask API and session handling
|   |-- vision.py              Combines behavior, expression, YOLO and scoring
|   |-- enforcement.py         Warning and blackout timer
|   `-- download_models.py     Downloads the model files
|-- behavior/                  Behavior detection (face absent, head pose, blinks)
|-- modules/
|   `-- gaze/                  Face and gaze analysis
|-- models/                    Model files (face_landmarker.task is tracked in Git)
|-- data/                      Generated CSV logs
|-- archive/                   Old backup files
|-- requirements.txt
`-- README.md
```

## Team and Modules

| Member | Module |
|---|---|
| Vaishnavi | Face and gaze analysis |
| Sanjana | Behavior detection, warning and blackout |
| Tripti | Backend and YOLO |
| Riya | Frontend |

## Setup

Use Python 3.9 to 3.12.

```bash
git clone https://github.com/SanaSharmaa0987/AI-Based-Interview-Proctoring-System.git
cd AI-Based-Interview-Proctoring-System

python -m venv venv
venv\Scripts\activate          # Windows (Command Prompt)
source venv/Scripts/activate   # Windows (Git Bash)
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Download the MediaPipe, FER+ and YOLO model files into `models/`:

```bash
python -m backend.download_models
```

The FER+ and YOLO files are large and ignored by Git, so this step is needed on every new machine.

## How to Run

Run all commands from the project root folder.

**Full system (backend and dashboard)**

```bash
python -m backend.app
```

1. Open `http://127.0.0.1:5000/` in the browser.
2. Fill in the candidate details and click Save Interview Details.
3. Click Enable Camera and allow camera permission.
4. Click Start Interview. Look at the camera for 2 seconds while the system calibrates.
5. Click Stop Interview when finished.

If port 5000 is already in use, set `PORT=5001` before the command and open `http://127.0.0.1:5001/`. Check `http://127.0.0.1:5000/api/health` to see which models or packages are missing.

**Standalone webcam tests (optional)**

These two scripts look for `face_landmarker.task` in the current folder. Copy it to the project root first:

```bash
cp models/face_landmarker.task .        # Mac/Linux and Git Bash
copy models\face_landmarker.task .      # Windows Command Prompt

python -m behavior.behavior_detection
python -m modules.gaze.gaze_detection
```

Press `q` to quit. Gaze results are saved in `gaze_results.csv` in the current folder. Only one program can use the webcam at a time, so close the others before testing.

## Suspicious Activity Score

The score is a simple, explainable heuristic. The highest matching rule is used.

| Signal | Score |
|---|---|
| Multiple faces | 90 |
| Face absent for 3 seconds or more | 80 |
| Phone detected | 80 |
| Book detected | 55 |
| Head turned away for 4 seconds or more | 45 |

Levels: Low (below 35), Elevated (35 to 69), High (70 and above). Expression output is not part of the score.

## Warning and Blackout

The server tracks how long suspicious activity continues without a break. A frame counts as suspicious when its score is 35 or higher.

| Stage | Trigger | What happens |
|---|---|---|
| Warning | 10 seconds of continuous suspicious activity | An orange warning banner appears at the top of the page |
| Blackout | 300 seconds (5 minutes) of continuous suspicious activity | A black screen covers the interview |

- One normal frame does not reset the timer, because object detection can miss a single frame. The timer resets only after 10 seconds of normal behavior.
- When normal behavior returns, the warning disappears.
- Once the blackout starts, it stays until the session ends.
- The timing runs on the server in `backend/enforcement.py`, so it cannot be changed from the browser.
- A web page cannot switch off a physical monitor, so the blackout is a full-screen black overlay.

The times are set at the top of `backend/enforcement.py` and can be changed with environment variables. For a quick test:

```bash
WARNING_SECONDS=10 BLACKOUT_SECONDS=20 python -m backend.app          # Mac/Linux and Git Bash
set WARNING_SECONDS=10 && set BLACKOUT_SECONDS=20 && python -m backend.app    # Windows Command Prompt
```

## Backend API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Check that the backend and models are ready |
| POST | `/api/interview/start` | Start a session (name, candidate_id and role are required) |
| POST | `/api/analysis/frame` | Analyze one JPEG frame (`frame` and `session_id` form fields) |
| GET | `/api/analysis/latest` | Get the latest results |
| POST | `/api/interview/stop` | Stop a session |

The frontend sends one resized camera frame per second while an interview is active.

Example response from `/api/analysis/latest`:

```json
{
  "gaze": { "value": "Center", "status": "Looking toward the screen" },
  "behavior": { "value": "Normal Posture", "status": "Behavior looks normal" },
  "expression": { "value": "Neutral", "confidence": 0.8 },
  "detected_objects": [],
  "suspicious_activity": { "value": "Low", "score": 0, "status": "No sustained signal detected." },
  "enforcement": { "level": "none", "sustained_seconds": 0.0, "message": "" }
}
```

`enforcement.level` is `none`, `warning` or `blackout`.

## Current Status

- [x] Frontend dashboard
- [x] Face detection and gaze estimation
- [x] Behavior detection module
- [x] Flask backend and dashboard integration
- [x] Facial expression CNN inference
- [x] YOLO phone and book detection
- [x] Combined suspicious activity score
- [x] Integration of all modules
- [x] Warning and blackout for sustained suspicious activity
- [ ] Demo video and final report

## Limitations

- Gaze and head pose are approximate estimates from face landmarks.
- Lighting, camera angle and glasses can affect accuracy.
- Expression output is an estimate and is not a measure of honesty.
- The blackout is a software overlay, not a hardware screen switch-off.
- The system should be used only with the candidate's consent and a clear privacy policy.

## Future Work

- More accurate head pose with `solvePnP`
- Include Left and Right gaze in the suspicious activity score
- Proctor control to resume a session after a blackout
- Validate phone and book detection under different camera conditions
- Session report with charts
- Fairness and accuracy testing on different users
