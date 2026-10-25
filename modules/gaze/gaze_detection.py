
import cv2
import mediapipe as mp
import csv
import os
import time
from datetime import datetime

MODEL_PATH = "face_landmarker.task"
CSV_FILE = "gaze_results.csv"

# MediaPipe Face Landmarker setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Prepare CSV file
file_exists = os.path.exists(CSV_FILE)

csv_file = open(CSV_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

if not file_exists or os.path.getsize(CSV_FILE) == 0:
    writer.writerow(["timestamp", "gaze_direction", "face_detected"])

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam not found. Check camera permission.")
    csv_file.close()
    raise SystemExit

print("Gaze detection started. Press Q to quit.")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        start_time = time.monotonic()
        last_log_time = 0
        gaze = "Unavailable"
        face_detected = False

        while True:
            success, frame = camera.read()

            if not success:
                print("Cannot read webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            face_detected = bool(result.face_landmarks)
            gaze = "Unavailable"

            if face_detected:
                points = result.face_landmarks[0]

                # Approximate left-eye iris position
                iris = points[468].x
                outer = points[33].x
                inner = points[133].x
                eye_width = abs(inner - outer)

                if eye_width > 0:
                    ratio = (
                        (iris - min(outer, inner)) / eye_width
                    )

                    if ratio < 0.35:
                        gaze = "Left"
                    elif ratio > 0.65:
                        gaze = "Right"
                    else:
                        gaze = "Center"

                height, width, _ = frame.shape

                for point in points:
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(
                        frame, (x, y), 1, (0, 255, 0), -1
                    )

            # Save one result per second
            elapsed = time.monotonic() - start_time

            if elapsed - last_log_time >= 1:
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    gaze,
                    face_detected
                ])
                csv_file.flush()
                last_log_time = elapsed
                print("Saved:", gaze, "| Face:", face_detected)

            face_text = (
                "Face: Detected"
                if face_detected
                else "Face: Not Detected"
            )

            cv2.putText(
                frame, face_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, "Gaze: " + gaze, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2
            )

            cv2.imshow("Gaze Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Gaze logging stopped. Results saved in", CSV_FILE)
