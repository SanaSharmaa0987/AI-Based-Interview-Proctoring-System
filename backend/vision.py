from pathlib import Path

import cv2
import numpy as np

from behavior.behavior_detection import BehaviorAnalyzer


EXPRESSION_LABELS = [
    "Neutral",
    "Happy",
    "Surprised",
    "Sad",
    "Angry",
    "Disgusted",
    "Fearful",
    "Contempt",
]
TARGET_OBJECTS = {"book", "cell phone"}


class InterviewAnalyzer:
    """Runs face behavior, FER+ expression, and COCO object detection on each frame."""

    def __init__(self, face_model: Path, expression_model: Path, yolo_model: Path):
        self.behavior = BehaviorAnalyzer(str(face_model))
        try:
            from ultralytics import YOLO

            self.expression_net = cv2.dnn.readNetFromONNX(str(expression_model))
            self.object_model = YOLO(str(yolo_model))
            names = self.object_model.names
            self.object_class_ids = [
                class_id for class_id, name in names.items()
                if name in TARGET_OBJECTS
            ]
            if not self.object_class_ids:
                raise RuntimeError("YOLO weights do not include phone and book classes")
        except Exception:
            self.behavior.close()
            raise

    def analyze(self, frame_bgr):
        behavior = self.behavior.analyze(frame_bgr)
        details = behavior.get("details", {})
        face_box = details.get("face_bbox")
        expression = self._classify_expression(frame_bgr, face_box)
        objects = self._detect_objects(frame_bgr)
        suspicious_activity = self._score_activity(behavior, objects)
        return {
            "gaze": self._gaze_result(details.get("gaze_direction", "Unavailable")),
            "behavior": behavior,
            "expression": expression,
            "detected_objects": objects,
            "suspicious_activity": suspicious_activity,
        }

    def close(self):
        self.behavior.close()

    def _classify_expression(self, frame_bgr, face_box):
        if not face_box:
            return {"value": "Unavailable", "status": "No single face to analyze"}
        x, y, width, height = face_box
        face = frame_bgr[y:y + height, x:x + width]
        if face.size == 0:
            return {"value": "Unavailable", "status": "Face crop is too small"}
        grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0))
        self.expression_net.setInput(blob)
        logits = self.expression_net.forward().reshape(-1).astype(np.float64)
        if logits.size != len(EXPRESSION_LABELS):
            return {"value": "Unavailable", "status": "Unexpected FER+ model output"}
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        label_index = int(np.argmax(probabilities))
        confidence = float(probabilities[label_index])
        return {
            "value": EXPRESSION_LABELS[label_index],
            "status": "CNN expression estimate; not a measure of honesty",
            "confidence": round(confidence, 3),
        }

    def _detect_objects(self, frame_bgr):
        result = self.object_model.predict(
            frame_bgr,
            classes=self.object_class_ids,
            conf=0.4,
            imgsz=416,
            device="cpu",
            verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = result.names[class_id]
            detections.append({
                "label": label,
                "confidence": round(float(box.conf.item()), 3),
            })
        return detections

    @staticmethod
    def _gaze_result(value):
        status = {
            "Center": "Looking toward the screen",
            "Left": "Looking left",
            "Right": "Looking right",
            "Unavailable": "Face or eye landmarks unavailable",
        }.get(value, "Gaze estimated from face landmarks")
        return {"value": value, "status": status}

    @staticmethod
    def _score_activity(behavior, objects):
        value = behavior.get("value", "")
        details = behavior.get("details", {})
        score = 0
        reasons = []
        if value == "Multiple Faces":
            score = max(score, 90)
            reasons.append("Multiple faces detected")
        elif value == "Face Absent":
            score = max(score, 80)
            reasons.append("Face absent for a sustained period")

        head_away = float(details.get("head_away_seconds", 0))
        if head_away >= 4:
            score = max(score, 45)
            reasons.append("Head turned away for a sustained period")

        object_labels = {item["label"] for item in objects}
        if "cell phone" in object_labels:
            score = max(score, 80)
            reasons.append("Phone detected in the camera view")
        if "book" in object_labels:
            score = max(score, 55)
            reasons.append("Book detected in the camera view")

        level = "High" if score >= 70 else "Elevated" if score >= 35 else "Low"
        status = "; ".join(reasons) if reasons else "No sustained signal detected"
        return {
            "value": level,
            "score": score,
            "status": status + ". Review signal only; not proof of misconduct.",
        }