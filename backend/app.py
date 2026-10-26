import csv
import importlib.util
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.vision import InterviewAnalyzer


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("FACE_LANDMARKER_MODEL", ROOT_DIR / "models" / "face_landmarker.task"))
EXPRESSION_MODEL_PATH = Path(os.environ.get("EXPRESSION_MODEL", ROOT_DIR / "models" / "emotion-ferplus-8.onnx"))
YOLO_MODEL_PATH = Path(os.environ.get("YOLO_MODEL", ROOT_DIR / "models" / "yolov8n.pt"))
RESULTS_PATH = ROOT_DIR / "data" / "session_results.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
MAX_FRAME_BYTES = 3 * 1024 * 1024

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES
CORS(app, resources={r"/api/*": {"origins": r"http://(localhost|127\.0\.0\.1)(:\d+)?$"}})

session_lock = threading.RLock()
session = {"active": False, "session_id": None, "candidate": None, "analyzer": None, "latest": None}


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


def empty_analysis():
    return {
        "gaze": {"value": "Unavailable", "status": "Waiting for a camera frame"},
        "behavior": {"value": "Waiting", "status": "No frame analyzed yet"},
        "expression": {"value": "Unavailable", "status": "No frame analyzed yet"},
        "detected_objects": [],
        "suspicious_activity": {"value": "Unknown", "status": "No analysis yet"},
    }


def model_status():
    return {
        "face_landmarker": MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 100_000,
        "expression_cnn": EXPRESSION_MODEL_PATH.is_file() and EXPRESSION_MODEL_PATH.stat().st_size > 1_000_000,
        "yolo_weights": YOLO_MODEL_PATH.is_file() and YOLO_MODEL_PATH.stat().st_size > 1_000_000,
        "ultralytics_installed": importlib.util.find_spec("ultralytics") is not None,
    }


def append_result(analysis):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if new_file:
            writer.writerow(["timestamp", "session_id", "gaze", "behavior", "suspicious_activity", "details"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            session["session_id"],
            analysis["gaze"]["value"],
            analysis["behavior"]["value"],
            analysis["suspicious_activity"]["value"],
            json.dumps({
                "behavior": analysis["behavior"].get("details", {}),
                "expression": analysis["expression"],
                "detected_objects": analysis["detected_objects"],
                "risk_score": analysis["suspicious_activity"].get("score"),
            }),
        ])


@app.get("/api/health")
def health():
    models = model_status()
    return jsonify({
        "status": "ok",
        "analysis_ready": all(models.values()),
        "models": models,
        "session_active": session["active"],
    })


@app.post("/api/interview/start")
def start_interview():
    payload = request.get_json(silent=True) or {}
    candidate = {key: str(payload.get(key, "")).strip() for key in ("name", "candidate_id", "role")}
    if not all(candidate.values()):
        return jsonify({"error": "name, candidate_id, and role are required"}), 400

    with session_lock:
        if session["active"]:
            return jsonify({"error": "An interview session is already active"}), 409
        models = model_status()
        missing = [name for name, ready in models.items() if not ready]
        if missing:
            return jsonify({
                "error": "Required model files or packages are missing. Run python -m backend.download_models and install requirements.txt.",
                "missing": missing,
            }), 503
        try:
            analyzer = InterviewAnalyzer(MODEL_PATH, EXPRESSION_MODEL_PATH, YOLO_MODEL_PATH)
        except Exception as error:
            app.logger.exception("Could not initialize the face analysis model")
            return jsonify({"error": f"Could not load face analysis model: {error}"}), 503

        session.update({"active": True, "session_id": str(uuid.uuid4()), "candidate": candidate, "analyzer": analyzer, "latest": empty_analysis()})
        return jsonify({"status": "started", "session_id": session["session_id"]})


@app.post("/api/analysis/frame")
def analyze_frame():
    frame_file = request.files.get("frame")
    requested_session = request.form.get("session_id", "")
    if frame_file is None:
        return jsonify({"error": "A JPEG frame is required"}), 400
    encoded_frame = frame_file.read(MAX_FRAME_BYTES + 1)
    if not encoded_frame:
        return jsonify({"error": "Frame is empty"}), 400
    if len(encoded_frame) > MAX_FRAME_BYTES:
        return jsonify({"error": "Frame exceeds the 3 MB limit"}), 413
    image = cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Frame is not a supported image"}), 400

    with session_lock:
        if not session["active"] or requested_session != session["session_id"]:
            return jsonify({"error": "No matching active interview session"}), 409
        try:
            analysis = session["analyzer"].analyze(image)
        except Exception:
            app.logger.exception("Frame analysis failed")
            return jsonify({"error": "Frame analysis failed"}), 500

        analysis["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        session["latest"] = analysis
        append_result(analysis)
        return jsonify(analysis)


@app.get("/api/analysis/latest")
def latest_analysis():
    with session_lock:
        return jsonify(session["latest"] or empty_analysis())


@app.post("/api/interview/stop")
def stop_interview():
    payload = request.get_json(silent=True) or {}
    requested_session = payload.get("session_id")
    with session_lock:
        if not session["active"]:
            return jsonify({"status": "stopped", "message": "No active session"})
        if requested_session and requested_session != session["session_id"]:
            return jsonify({"error": "Session ID does not match"}), 409
        analyzer = session["analyzer"]
        session.update({"active": False, "session_id": None, "candidate": None, "analyzer": None})
        if analyzer is not None:
            analyzer.close()
        return jsonify({"status": "stopped"})


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request exceeds the 3 MB limit"}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)