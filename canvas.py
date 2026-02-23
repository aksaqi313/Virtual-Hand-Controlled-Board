# -*- coding: utf-8 -*-
"""
canvas.py
---------
Manages the persistent drawing canvas that overlays the camera feed.
"""

import numpy as np
import cv2


class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_point = None

    def draw(self, point, color, thickness=8):
        """Draw a smooth stroke between the previous and current point."""
        if self.prev_point is None:
            self.prev_point = point
        cv2.line(self.layer, self.prev_point, point, color, thickness, lineType=cv2.LINE_AA)
        self.prev_point = point

    def erase(self, point, size=50):
        """Erase a circular region around the given point."""
        cv2.circle(self.layer, point, size, (0, 0, 0), -1)
        self.prev_point = point

    def reset_stroke(self):
        """Call when the drawing finger lifts to break stroke continuity."""
        self.prev_point = None

    def clear(self):
        """Wipe the entire canvas."""
        self.layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.prev_point = None

    def save(self, path="drawing.png"):
        """Save the canvas layer as an image."""
        cv2.imwrite(path, self.layer)

    def blend(self, background, opacity=0.85):
        """
        Blend the canvas drawing over the background camera frame.
        Only painted (non-black) pixels are overlaid.
        """
        mask = cv2.cvtColor(self.layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(background, background, mask=mask_inv)
        fg = cv2.addWeighted(self.layer, opacity, np.zeros_like(self.layer), 0, 0)
        fg = cv2.bitwise_and(fg, fg, mask=mask)

        return cv2.add(bg, fg)
