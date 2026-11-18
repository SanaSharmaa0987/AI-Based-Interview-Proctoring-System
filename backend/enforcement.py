"""
enforcement.py
Tracks how long suspicious activity has been going on without a break,
and decides when to show a warning or black out the screen.

Defaults:  warning after 2 minutes, blackout after 5 minutes.
For quick testing, set environment variables before starting the server:
    WARNING_SECONDS=10 BLACKOUT_SECONDS=20 python -m backend.app
"""

import os
import time

WARNING_SECONDS = float(os.environ.get("WARNING_SECONDS", 120))    # 2 minutes
BLACKOUT_SECONDS = float(os.environ.get("BLACKOUT_SECONDS", 300))  # 5 minutes
RESET_GRACE_SECONDS = float(os.environ.get("RESET_GRACE_SECONDS", 10))
SUSPICIOUS_SCORE = 35  # same cut-off the app uses for the "Elevated" level


class SustainedActivityTracker:
    """Create one per interview session and call update() for every frame."""

    def __init__(self):
        self.started_at = None          # when the current suspicious streak began
        self.last_suspicious_at = None  # last frame that was suspicious
        self.blackout_latched = False   # once True, stays True until session ends

    def update(self, score, now=None):
        now = time.monotonic() if now is None else now

        if score >= SUSPICIOUS_SCORE:
            if self.started_at is None:
                self.started_at = now
            self.last_suspicious_at = now
        elif self.started_at is not None:
            # One clean frame does not end the streak (detections can flicker).
            # The streak resets only after a short, steady period of normal behavior.
            if now - self.last_suspicious_at >= RESET_GRACE_SECONDS:
                self.started_at = None
                self.last_suspicious_at = None

        sustained = 0.0 if self.started_at is None else now - self.started_at

        if self.blackout_latched or sustained >= BLACKOUT_SECONDS:
            self.blackout_latched = True
            level, message = "blackout", "Screen turned off due to continuous suspicious activity."
        elif sustained >= WARNING_SECONDS:
            level = "warning"
            left = max(0, int(BLACKOUT_SECONDS - sustained))
            message = f"Warning: suspicious activity detected. Screen turns off in {left} seconds."
        else:
            level, message = "none", ""

        return {
            "level": level,
            "sustained_seconds": round(sustained, 1),
            "message": message,
        }
