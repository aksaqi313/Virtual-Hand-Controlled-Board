# -*- coding: utf-8 -*-
"""
hand_tracker.py
---------------
Handles hand detection and landmark extraction using the
MediaPipe Tasks API (mediapipe >= 0.10.20).
"""

import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"

# Connection pairs between landmarks for drawing skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._detection_result = None
        self._timestamp_ms = 0

        # Landmark indices
        self.WRIST      = 0
        self.THUMB_TIP  = 4
        self.INDEX_TIP  = 8
        self.INDEX_PIP  = 6
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        self.RING_TIP   = 16
        self.RING_PIP   = 14
        self.PINKY_TIP  = 20
        self.PINKY_PIP  = 18

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def find_hands(self, frame, draw=True):
        """Run inference on the frame; optionally draw skeleton. Returns frame."""
        self._timestamp_ms += 33  # simulate ~30 fps timestamps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._detection_result = self.landmarker.detect_for_video(
            mp_image, self._timestamp_ms
        )

        if draw and self._detection_result and self._detection_result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand in self._detection_result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                # Draw connections
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 120), 2, cv2.LINE_AA)

                # Draw landmark dots
                for i, (x, y) in enumerate(pts):
                    r = 5 if i in (4, 8, 12, 16, 20) else 3
                    cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), r, (0, 160, 100), 1, cv2.LINE_AA)

        return frame

    def get_landmarks(self, frame, hand_index=0):
        """Return pixel-space landmark list [(x, y), ...] for a given hand."""
        h, w = frame.shape[:2]
        if (
            self._detection_result is None
            or not self._detection_result.hand_landmarks
            or hand_index >= len(self._detection_result.hand_landmarks)
        ):
            return []
        hand = self._detection_result.hand_landmarks[hand_index]
        return [(int(lm.x * w), int(lm.y * h)) for lm in hand]

    # ------------------------------------------------------------------
    # Gesture helpers
    # ------------------------------------------------------------------

    def fingers_up(self, landmarks):
        """Return [thumb, index, middle, ring, pinky] — 1=up, 0=down."""
        if not landmarks:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb: compare x of tip vs knuckle (mirrored feed)
        if landmarks[self.THUMB_TIP][0] < landmarks[self.THUMB_TIP - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers: tip.y < pip.y  →  finger is raised
        for tip, pip in [
            (self.INDEX_TIP,  self.INDEX_PIP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP,   self.RING_PIP),
            (self.PINKY_TIP,  self.PINKY_PIP),
        ]:
            fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

        return fingers

    def distance(self, p1, p2):
        """Euclidean distance between two landmark points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def get_gesture(self, landmarks):
        """
        Classify gesture:
          'draw'   – only index finger raised
          'erase'  – index + middle raised (peace ✌)
          'select' – all four fingers raised (open palm)
          'idle'   – anything else
        """
        if not landmarks:
            return "idle"

        f = self.fingers_up(landmarks)

        if f == [0, 1, 0, 0, 0]:
            return "draw"
        elif f == [0, 1, 1, 0, 0]:
            return "erase"
        elif f[1] == 1 and f[2] == 1 and f[3] == 1 and f[4] == 1:
            return "select"
        else:
            return "idle"
