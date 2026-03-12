# -*- coding: utf-8 -*-
"""
ui.py
-----
Renders the heads-up toolbar and handles hit-testing for toolbar actions.
"""

import cv2
import numpy as np


# ── Toolbar layout constants ───────────────────────────────────────────────────
TOOLBAR_H = 90          # pixel height of the toolbar strip
SWATCH_R  = 22          # radius of each colour circle
BTN_W     = 90          # width of rectangular buttons
BTN_H     = 50          # height of rectangular buttons

# Colour palette  (BGR)
PALETTE = [
    ("White",   (255, 255, 255)),
    ("Yellow",  (0,   220, 255)),
    ("Cyan",    (255, 220,   0)),
    ("Green",   (50,  205,  50)),
    ("Pink",    (180,  90, 255)),
    ("Red",     (50,   50, 220)),
    ("Blue",    (220,  80,  50)),
    ("Orange",  (0,   140, 255)),
]

BRUSH_SIZES = [4, 8, 14, 22]   # thickness options


class Toolbar:
    def __init__(self, width):
        self.width = width
        self.active_color_idx = 0
        self.active_size_idx  = 1   # default: 8
        self.eraser_mode      = False
        self._build_layout()

    @property
    def color(self):
        return PALETTE[self.active_color_idx][1]

    @property
    def brush_size(self):
        return BRUSH_SIZES[self.active_size_idx]

    # ------------------------------------------------------------------
    # Layout builder – called once; stores hit-boxes
    # ------------------------------------------------------------------

    def _build_layout(self):
        self.color_boxes  = []   # list of (cx, cy, idx)
        self.size_boxes   = []   # list of (x1, y1, x2, y2, idx)
        self.eraser_box   = None # (x1, y1, x2, y2)
        self.clear_box    = None
        self.save_box     = None

        cx = 30
        cy = TOOLBAR_H // 2

        # ── Colour swatches ──────────────────────────────────────────
        for i in range(len(PALETTE)):
            self.color_boxes.append((cx, cy, i))
            cx += SWATCH_R * 2 + 12

        cx += 10  # small gap before size buttons

        # ── Brush size dots ──────────────────────────────────────────
        for i, s in enumerate(BRUSH_SIZES):
            self.size_boxes.append((cx, cy - BTN_H // 2, cx + BTN_W, cy + BTN_H // 2, i))
            cx += BTN_W + 8

        cx += 10

        # ── Eraser button ─────────────────────────────────────────────
        self.eraser_box = (cx, cy - BTN_H // 2, cx + BTN_W, cy + BTN_H // 2)
        cx += BTN_W + 8

        # ── Clear button ──────────────────────────────────────────────
        self.clear_box = (cx, cy - BTN_H // 2, cx + BTN_W, cy + BTN_H // 2)
        cx += BTN_W + 8

        # ── Save button ───────────────────────────────────────────────
        self.save_box = (cx, cy - BTN_H // 2, cx + BTN_W, cy + BTN_H // 2)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, frame):
        """Paint the toolbar strip onto the given frame (in-place)."""
        # Frosted-glass background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, TOOLBAR_H), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Bottom border line
        cv2.line(frame, (0, TOOLBAR_H), (self.width, TOOLBAR_H), (80, 80, 120), 2)

        # ── Colour swatches ──────────────────────────────────────────
        for i, (name, bgr) in enumerate(PALETTE):
            cx, cy, _ = self.color_boxes[i]
            selected = (i == self.active_color_idx) and not self.eraser_mode

            # glow ring for active colour
            if selected:
                cv2.circle(frame, (cx, cy), SWATCH_R + 5, bgr, 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), SWATCH_R + 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.circle(frame, (cx, cy), SWATCH_R, bgr, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), SWATCH_R, (60, 60, 70), 1, lineType=cv2.LINE_AA)

        # ── Brush size buttons ────────────────────────────────────────
        for i, (x1, y1, x2, y2, _) in enumerate(self.size_boxes):
            selected = (i == self.active_size_idx)
            color_bg = (60, 60, 100) if selected else (35, 35, 50)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bg, -1, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 180), 1, lineType=cv2.LINE_AA)

            # indicator dot
            dot_r = BRUSH_SIZES[i] // 2 + 1
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.circle(frame, (mid_x, mid_y - 4), dot_r, (255, 255, 255), -1)
            cv2.putText(frame, str(BRUSH_SIZES[i]), (mid_x - 5, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 200), 1, cv2.LINE_AA)

        # ── Eraser ───────────────────────────────────────────────────
        x1, y1, x2, y2 = self.eraser_box
        ec = (60, 60, 100) if self.eraser_mode else (35, 35, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ec, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 180), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "ERASE", (x1 + 10, (y1 + y2) // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1, cv2.LINE_AA)

        # ── Clear ─────────────────────────────────────────────────────
        x1, y1, x2, y2 = self.clear_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 50), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 180), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "CLEAR", (x1 + 10, (y1 + y2) // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1, cv2.LINE_AA)

        # ── Save ──────────────────────────────────────────────────────
        x1, y1, x2, y2 = self.save_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 80, 30), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 160, 50), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "SAVE", (x1 + 14, (y1 + y2) // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Hit-testing 
    # ------------------------------------------------------------------

    def check_hit(self, point):
        """
        Test whether a given point lands on a toolbar element.
        Returns one of: 'color', 'size', 'erase', 'clear', 'save', or None.
        Also updates the active colour/size on hit.
        """
        if point is None:
            return None
        x, y = point
        if y > TOOLBAR_H:
            return None

        # Colour swatches
        for cx, cy, i in self.color_boxes:
            if (x - cx) ** 2 + (y - cy) ** 2 <= (SWATCH_R + 5) ** 2:
                self.active_color_idx = i
                self.eraser_mode = False
                return "color"

        # Brush size
        for x1, y1, x2, y2, i in self.size_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.active_size_idx = i
                return "size"

        # Eraser
        x1, y1, x2, y2 = self.eraser_box
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.eraser_mode = not self.eraser_mode
            return "erase"

        # Clear
        x1, y1, x2, y2 = self.clear_box
        if x1 <= x <= x2 and y1 <= y <= y2:
            return "clear"

        # Save
        x1, y1, x2, y2 = self.save_box
        if x1 <= x <= x2 and y1 <= y <= y2:
            return "save"

        return None
