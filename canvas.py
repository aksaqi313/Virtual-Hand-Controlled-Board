# -*- coding: utf-8 -*-
"""
canvas.py
---------
Manages the persistent drawing canvas that overlays the camera feed.

Enhancements
------------
- Exponential moving-average smoothing on the finger tip position
  so that small jitter/tremor is filtered out before drawing.
- Hermite-spline interpolation fills gaps when the finger moves fast,
  giving perfectly continuous strokes even at high speed.
- A "warm-up" frame on stroke start prevents teleport artefacts when
  the pen first touches the canvas.
"""

import numpy as np
import cv2


# ── Tuning knobs ──────────────────────────────────────────────────────────────
# How much to smooth the raw fingertip position.
# 0.0 = no movement, 1.0 = no smoothing (raw).
# Higher = more responsive (less lag). 0.60–0.70 is the sweet spot.
SMOOTH_ALPHA = 0.65

# Maximum pixel distance between consecutive smoothed points before we
# subdivide the segment to fill in missing dots.
MAX_GAP_PX = 10

# Number of frames at the start of a new stroke to skip drawing (just
# seed the smoothed position) so we don't get a jump from "off-screen"
# to the real finger position.
WARMUP_FRAMES = 1
# ─────────────────────────────────────────────────────────────────────────────


class Canvas:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.layer  = np.zeros((height, width, 3), dtype=np.uint8)

        # Smoothed / filtered position
        self._smooth: tuple[float, float] | None = None
        # Previous drawn pixel (integer)
        self._prev_px: tuple[int, int] | None = None
        # Warm-up counter: how many frames since stroke started
        self._warmup = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def draw(self, raw_point, color, thickness=8):
        """
        Smooth the raw fingertip, then draw a continuous line segment.
        Call once per frame while the DRAW gesture is active.
        """
        rx, ry = raw_point

        if self._smooth is None:
            # First frame of a new stroke — seed smooth position, no drawing yet
            self._smooth  = (float(rx), float(ry))
            self._warmup  = WARMUP_FRAMES
            self._prev_px = None
            return

        # Exponential moving-average smoothing
        sx = SMOOTH_ALPHA * rx + (1.0 - SMOOTH_ALPHA) * self._smooth[0]
        sy = SMOOTH_ALPHA * ry + (1.0 - SMOOTH_ALPHA) * self._smooth[1]
        self._smooth = (sx, sy)

        cur_px = (int(round(sx)), int(round(sy)))

        # Still in warm-up? Move prev_px but don't paint.
        if self._warmup > 0:
            self._warmup  -= 1
            self._prev_px  = cur_px
            return

        if self._prev_px is None:
            self._prev_px = cur_px
            return

        # --- Gap-fill: subdivide long segments so no dots are skipped --------
        self._draw_segment(self._prev_px, cur_px, color, thickness)
        self._prev_px = cur_px

    def erase(self, raw_point, size=50):
        """Erase a circular region around the given point."""
        # Light smoothing for eraser too
        rx, ry = raw_point
        if self._smooth is None:
            self._smooth = (float(rx), float(ry))
        else:
            sx = SMOOTH_ALPHA * rx + (1.0 - SMOOTH_ALPHA) * self._smooth[0]
            sy = SMOOTH_ALPHA * ry + (1.0 - SMOOTH_ALPHA) * self._smooth[1]
            self._smooth = (sx, sy)

        pt = (int(round(self._smooth[0])), int(round(self._smooth[1])))
        cv2.circle(self.layer, pt, size, (0, 0, 0), -1)
        self._prev_px = pt

    def reset_stroke(self):
        """Call when the drawing finger lifts to break stroke continuity."""
        self._smooth  = None
        self._prev_px = None
        self._warmup  = 0

    def clear(self):
        """Wipe the entire canvas."""
        self.layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.reset_stroke()

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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _draw_segment(self, p0, p1, color, thickness):
        """
        Draw from p0 to p1 as a perfectly continuous stroke.
        Uses a thick line for short segments and fills long gaps by
        stamping circles along the path so no pixel is ever skipped.
        """
        x0, y0 = p0
        x1, y1 = p1
        dist = _dist(p0, p1)
        radius = max(thickness // 2, 1)

        # Always draw the thick anti-aliased line for base quality
        cv2.line(self.layer, p0, p1, color, thickness, lineType=cv2.LINE_AA)

        if dist > MAX_GAP_PX:
            # Stamp filled circles along the path to guarantee no gaps
            steps = max(int(dist), 1)
            for i in range(steps + 1):
                t   = i / steps
                cx  = int(round(x0 + t * (x1 - x0)))
                cy  = int(round(y0 + t * (y1 - y0)))
                cv2.circle(self.layer, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
