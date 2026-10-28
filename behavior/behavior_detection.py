"""
behavior_detection.py
Behavior Detection module for the AI Interview Analysis project.

What it detects (simple and explainable):
  1. Face absent         - no face in the frame for some seconds
  2. Multiple faces      - more than one person in the frame
  3. Head turned         - left / right (yaw)
  4. Head up / down      - looking up or down (pitch)
  5. Blink rate          - blinks per minute (too high or too low)

How the backend should use it:
    from behavior_detection import BehaviorAnalyzer
    analyzer = BehaviorAnalyzer()
    result = analyzer.analyze(frame_bgr)   # returns a dict
    # {"value": "Head Turned Left", "status": "Warning: ...", "details": {...}}

The "value" and "status" keys match what the frontend cards expect.

"""

import time
from collections import deque

import cv2
import mediapipe as mp

MODEL_PATH = "face_landmarker.task"  # same model file used in the gaze module

# ---------------- Settings (change these to tune the module) ----------------
CALIBRATION_FRAMES = 30       # frames used to learn the person's normal pose
YAW_LIMIT = 0.15              # allowed change in left/right head position
PITCH_LIMIT = 0.12            # allowed change in up/down head position
EAR_BLINK_THRESHOLD = 0.21    # eye aspect ratio below this = eye closed
FACE_ABSENT_SECONDS = 3       # warn if no face for this long
HEAD_AWAY_SECONDS = 4         # warn if head pose is off for this long
BLINK_WINDOW_SECONDS = 60     # blink rate is counted over this window
BLINK_LOW = 5                 # blinks per minute (too low)
BLINK_HIGH = 40               # blinks per minute (too high)

# ---------------- MediaPipe landmark indexes ----------------
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Eye points in this order: [corner, top1, top2, corner, bottom2, bottom1]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _dist(a, b):
    """Distance between two landmarks."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def eye_aspect_ratio(points, eye):
    """Eye Aspect Ratio (EAR). Low value means the eye is closed."""
    p1, p2, p3, p4, p5, p6 = [points[i] for i in eye]
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 1.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def head_pose_ratios(points):
    """
    Returns (yaw_ratio, pitch_ratio).
    yaw_ratio   ~0.5 when facing the camera, goes to 0 or 1 when turned.
    pitch_ratio changes when the head goes up or down.
    """
    nose = points[NOSE_TIP]
    left, right = points[LEFT_CHEEK], points[RIGHT_CHEEK]
    top, chin = points[FOREHEAD], points[CHIN]

    face_width = right.x - left.x
    face_height = chin.y - top.y
    if face_width == 0 or face_height == 0:
        return 0.5, 0.5

    yaw = (nose.x - left.x) / face_width
    pitch = (nose.y - top.y) / face_height
    return yaw, pitch


def gaze_direction(points):
    """Estimate horizontal gaze from the left iris and eye corners."""
    if len(points) <= 468:
        return "Unavailable"
    iris = points[468].x
    outer = points[33].x
    inner = points[133].x
    eye_width = abs(inner - outer)
    if eye_width == 0:
        return "Unavailable"
    ratio = (iris - min(outer, inner)) / eye_width
    if ratio < 0.35:
        return "Left"
    if ratio > 0.65:
        return "Right"
    return "Center"


class BehaviorAnalyzer:
    """Keeps state across frames, so create it once and call analyze() per frame."""

    def __init__(self, model_path=MODEL_PATH):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,  # 2 so we can detect "multiple faces"
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.start_time = time.monotonic()
        self.reset()

    def reset(self):
        """Call this when a new interview starts."""
        self.calib_yaw = []
        self.calib_pitch = []
        self.base_yaw = None
        self.base_pitch = None
        self.face_missing_since = None
        self.head_away_since = None
        self.eye_closed = False
        self.blink_times = deque()

    def close(self):
        self.landmarker.close()

    # ------------------------------------------------------------------
    def analyze(self, frame_bgr):
        """Analyze one webcam frame (OpenCV BGR image). Returns a dict."""
        now = time.monotonic()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((now - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        faces = result.face_landmarks
        details = {
            "faces": len(faces),
            "blinks_per_min": self._blink_rate(now),
            "gaze_direction": "Unavailable",
        }

        # 1) No face
        if not faces:
            if self.face_missing_since is None:
                self.face_missing_since = now
            missing = now - self.face_missing_since
            details["face_missing_seconds"] = round(missing, 1)
            self.head_away_since = None
            if missing >= FACE_ABSENT_SECONDS:
                return self._result(
                    "Face Absent",
                    f"Alert: no face for {int(missing)} seconds",
                    details,
                )
            return self._result("Checking", "Face not visible", details)
        self.face_missing_since = None

        # 2) More than one face
        if len(faces) > 1:
            return self._result(
                "Multiple Faces", "Alert: more than one person detected", details
            )

        points = faces[0]
        frame_height, frame_width = frame_bgr.shape[:2]
        left = max(0, int(min(point.x for point in points) * frame_width))
        top = max(0, int(min(point.y for point in points) * frame_height))
        right = min(frame_width, int(max(point.x for point in points) * frame_width))
        bottom = min(frame_height, int(max(point.y for point in points) * frame_height))
        details["face_bbox"] = [left, top, right - left, bottom - top]
        details["gaze_direction"] = gaze_direction(points)
        yaw, pitch = head_pose_ratios(points)

        # 3) Calibration: learn the normal pose first
        if self.base_yaw is None:
            self.calib_yaw.append(yaw)
            self.calib_pitch.append(pitch)
            if len(self.calib_yaw) >= CALIBRATION_FRAMES:
                self.base_yaw = sum(self.calib_yaw) / len(self.calib_yaw)
                self.base_pitch = sum(self.calib_pitch) / len(self.calib_pitch)
            return self._result(
                "Calibrating", "Look at the camera for 2 seconds", details
            )

        # 4) Blink tracking
        ear = (
            eye_aspect_ratio(points, LEFT_EYE) + eye_aspect_ratio(points, RIGHT_EYE)
        ) / 2
        self._update_blinks(ear, now)
        details["blinks_per_min"] = self._blink_rate(now)
        details["ear"] = round(ear, 3)

        # 5) Head pose compared with the baseline
        d_yaw = yaw - self.base_yaw
        d_pitch = pitch - self.base_pitch
        details["yaw_change"] = round(d_yaw, 3)
        details["pitch_change"] = round(d_pitch, 3)

        pose = None
        if d_yaw < -YAW_LIMIT:
            pose = "Head Turned Left"
        elif d_yaw > YAW_LIMIT:
            pose = "Head Turned Right"
        elif d_pitch > PITCH_LIMIT:
            pose = "Looking Down"
        elif d_pitch < -PITCH_LIMIT:
            pose = "Looking Up"

        if pose:
            if self.head_away_since is None:
                self.head_away_since = now
            away = now - self.head_away_since
            details["head_away_seconds"] = round(away, 1)
            if away >= HEAD_AWAY_SECONDS:
                return self._result(
                    pose, f"Warning: head away for {int(away)} seconds", details
                )
            return self._result(pose, "Brief movement", details)
        self.head_away_since = None

        # 6) Blink rate (only after 20 seconds, so the number is stable)
        rate = details["blinks_per_min"]
        if now - self.start_time > 20:
            if rate > BLINK_HIGH:
                return self._result("Normal Posture", "Note: high blink rate", details)
            if rate < BLINK_LOW:
                return self._result("Normal Posture", "Note: low blink rate", details)

        return self._result("Normal Posture", "Behavior looks normal", details)

    # ------------------------------------------------------------------
    def _update_blinks(self, ear, now):
        """A blink is counted when the eye goes closed and then opens again."""
        if ear < EAR_BLINK_THRESHOLD:
            self.eye_closed = True
        elif self.eye_closed:
            self.eye_closed = False
            self.blink_times.append(now)

    def _blink_rate(self, now):
        while self.blink_times and now - self.blink_times[0] > BLINK_WINDOW_SECONDS:
            self.blink_times.popleft()
        window = min(BLINK_WINDOW_SECONDS, max(now - self.start_time, 1))
        return round(len(self.blink_times) * 60 / window, 1)

    @staticmethod
    def _result(value, status, details):
        return {"value": value, "status": status, "details": details}


# ---------------- Webcam test ----------------
if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Webcam not found. Check camera permission.")
        raise SystemExit

    analyzer = BehaviorAnalyzer()
    print("Behavior detection started. Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Cannot read webcam.")
                break

            frame = cv2.flip(frame, 1)
            output = analyzer.analyze(frame)

            cv2.putText(frame, output["value"], (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, output["status"], (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Behavior Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        analyzer.close()
        camera.release()
        cv2.destroyAllWindows()
