# -*- coding: utf-8 -*-
"""
main.py
-------
Virtual Hand-Controlled Board
==============================
Draw on a virtual whiteboard using only hand gestures via your webcam.

Gestures
--------
  ☝  Index finger only  → DRAW
  ✌  Index + Middle     → ERASE
  🖐  Open palm          → SELECT toolbar items (hover ~12 frames)
  ✊  Fist               → IDLE / lift pen

Keyboard shortcuts
------------------
  Q or ESC   → Quit
  C          → Clear canvas
  S          → Save drawing as PNG
  +/-        → Increase / decrease brush size
"""

import cv2
import numpy as np
import time
import datetime

from hand_tracker import HandTracker
from canvas import Canvas
from ui import Toolbar, TOOLBAR_H

# ── Configuration ─────────────────────────────────────────────────────────────
CAM_INDEX            = 0
FRAME_W              = 1280
FRAME_H              = 720
TARGET_FPS           = 30
TOOLBAR_HOVER_FRAMES = 12
SPLASH_DURATION      = 3.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_status_bar(frame, gesture, color, brush_size, eraser_mode, fps, save_msg):
    h, w = frame.shape[:2]
    bar_y = h - 30
    cv2.rectangle(frame, (0, bar_y), (w, h), (15, 15, 25), -1)

    mode_text    = "ERASER" if eraser_mode else f"Draw  size {brush_size}"
    gesture_text = f"Gesture: {gesture.upper()}"
    fps_text     = f"FPS: {int(fps)}"

    cv2.putText(frame, mode_text,    (12, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color if not eraser_mode else (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, gesture_text, (300, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, fps_text,     (w - 100, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 180, 120), 1, cv2.LINE_AA)
    if save_msg:
        cv2.putText(frame, save_msg, (w // 2 - 200, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 120), 1, cv2.LINE_AA)


def draw_cursor(frame, point, gesture, color, brush_size, eraser_mode):
    if point is None:
        return
    x, y = point
    if eraser_mode or gesture == "erase":
        cv2.circle(frame, (x, y), 26, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4,  (255, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), brush_size // 2 + 4, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3,  (255, 255, 255), -1, cv2.LINE_AA)


def draw_splash(frame, alpha):
    if alpha <= 0:
        return frame
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (w // 4, h // 5), (3 * w // 4, 4 * h // 5), (15, 15, 30), -1)

    lines = [
        ("Virtual Hand-Controlled Board", 0.95, (80, 200, 255)),
        ("",                               0.5,  (200, 200, 200)),
        ("  Index finger  ->  DRAW",       0.65, (255, 255, 255)),
        ("  Peace sign    ->  ERASE",      0.65, (220, 220, 255)),
        ("  Open palm     ->  SELECT toolbar", 0.65, (220, 220, 255)),
        ("  Fist          ->  IDLE",       0.65, (180, 180, 220)),
        ("",                               0.5,  (200, 200, 200)),
        ("  Q/ESC  Quit    C  Clear    S  Save",  0.55, (160, 160, 200)),
        ("  +/-  Change brush size",       0.55, (160, 160, 200)),
    ]

    y = h // 5 + 50
    for text, scale, col in lines:
        thickness = 2 if scale >= 0.8 else 1
        cv2.putText(overlay, text, (w // 4 + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness, cv2.LINE_AA)
        y += int(45 * scale) + 10

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_hover_progress(frame, x, y, progress):
    angle = int(360 * progress)
    cv2.ellipse(frame, (x, y), (32, 32), -90, 0, angle, (100, 200, 255), 3, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Try changing CAM_INDEX in main.py.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened at {actual_w}x{actual_h}")

    tracker = HandTracker(max_hands=1)
    canvas  = Canvas(actual_w, actual_h)
    toolbar = Toolbar(actual_w)

    prev_time    = time.time()
    fps          = 0.0
    hover_count  = 0
    splash_start = time.time()
    save_msg     = ""
    save_msg_end = 0.0

    cv2.namedWindow("Virtual Hand-Controlled Board", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Hand-Controlled Board", actual_w, actual_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ── Hand detection ────────────────────────────────────────────
        frame     = tracker.find_hands(frame, draw=True)
        landmarks = tracker.get_landmarks(frame)
        gesture   = tracker.get_gesture(landmarks)
        index_tip = landmarks[tracker.INDEX_TIP] if landmarks else None

        # ── FPS ───────────────────────────────────────────────────────
        now      = time.time()
        fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── Toolbar hover (open-palm gesture) ─────────────────────────
        action = None
        if gesture == "select" and index_tip and index_tip[1] < TOOLBAR_H:
            hover_count += 1
            progress = min(hover_count / TOOLBAR_HOVER_FRAMES, 1.0)
            draw_hover_progress(frame, *index_tip, progress)
            if hover_count >= TOOLBAR_HOVER_FRAMES:
                action = toolbar.check_hit(index_tip)
                hover_count = 0
                canvas.reset_stroke()
        else:
            hover_count = 0

        if action == "clear":
            canvas.clear()
        elif action == "save":
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"drawing_{ts}.png"
            canvas.save(path)
            save_msg     = f"Saved -> {path}"
            save_msg_end = time.time() + 3.0

        # ── Draw / Erase ──────────────────────────────────────────────
        if index_tip and index_tip[1] > TOOLBAR_H:
            if gesture == "draw":
                if toolbar.eraser_mode:
                    canvas.erase(index_tip, size=26)
                else:
                    canvas.draw(index_tip, toolbar.color, toolbar.brush_size)
            elif gesture == "erase":
                canvas.erase(index_tip, size=26)
            else:
                canvas.reset_stroke()
        else:
            canvas.reset_stroke()

        # ── Compose & display ─────────────────────────────────────────
        output = canvas.blend(frame)
        output = toolbar.draw(output)
        draw_cursor(output, index_tip, gesture, toolbar.color, toolbar.brush_size, toolbar.eraser_mode)
        draw_status_bar(output, gesture, toolbar.color, toolbar.brush_size, toolbar.eraser_mode, fps,
                        save_msg if time.time() < save_msg_end else "")

        elapsed = time.time() - splash_start
        if elapsed < SPLASH_DURATION:
            alpha = max(0.0, 1.0 - elapsed / SPLASH_DURATION)
            draw_splash(output, alpha)

        cv2.imshow("Virtual Hand-Controlled Board", output)

        # ── Keyboard shortcuts ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            canvas.clear()
        elif key == ord('s'):
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"drawing_{ts}.png"
            canvas.save(path)
            save_msg     = f"Saved -> {path}"
            save_msg_end = time.time() + 3.0
        elif key in (ord('+'), ord('=')):
            toolbar.active_size_idx = min(toolbar.active_size_idx + 1, len(toolbar.size_boxes) - 1)
        elif key == ord('-'):
            toolbar.active_size_idx = max(toolbar.active_size_idx - 1, 0)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
