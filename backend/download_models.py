from pathlib import Path
from urllib.request import urlretrieve


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "emotion-ferplus-8.onnx": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
}
MINIMUM_SIZES = {
    "face_landmarker.task": 100_000,
    "emotion-ferplus-8.onnx": 1_000_000,
    "yolov8n.pt": 1_000_000,
}


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        destination = MODEL_DIR / filename
        temporary = destination.with_suffix(destination.suffix + ".download")
        print(f"Downloading {filename}...")
        try:
            urlretrieve(url, temporary)
            if temporary.stat().st_size < MINIMUM_SIZES[filename]:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {temporary.stat().st_size} bytes")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()