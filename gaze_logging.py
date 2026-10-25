
import csv
import os
from datetime import datetime

CSV_FILE = "gaze_results.csv"

def save_gaze_result(gaze, face_detected):
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "gaze_direction",
                "face_detected"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            gaze,
            face_detected
        ])

    print("Gaze result saved.")
    
if __name__ == "__main__":
    save_gaze_result("Center", True)
    print("Test complete. Check gaze_results.csv")
