"""
Whiteboard Detector Module.

Detects the whiteboard region from video frames using HSV color
segmentation (white/bright region detection) and morphological
operations. This approach works much better than edge-based
contour detection when the board fills most of the frame or
doesn't have clean rectangular edges.
"""

import cv2
import numpy as np
import config


class WhiteboardDetector:
    """
    Detects and crops the whiteboard region using color segmentation.

    Strategy:
    1. Convert to HSV color space
    2. Threshold for white/bright, low-saturation regions (whiteboard)
    3. Morphological cleanup to get a solid mask
    4. Find the bounding rectangle of the largest white region
    5. Crop and return

    The detected region is cached since the board doesn't move.
    """

    def __init__(self):
        self._cached_bbox = None      # (x, y, w, h)
        self._cache_counter = 0
        self._frame_shape = None

    def detect(self, frame):
        """
        Detect the whiteboard and return a cropped image.

        Parameters
        ----------
        frame : np.ndarray
            BGR video frame.

        Returns
        -------
        np.ndarray or None
            Cropped whiteboard image, or None if no board detected.
        """
        self._frame_shape = frame.shape[:2]

        # Use cached bbox if available and not expired
        if self._cached_bbox is not None and self._cache_counter < config.BOARD_CACHE_FRAMES:
            self._cache_counter += 1
            return self._crop(frame, self._cached_bbox)

        # Detect fresh
        bbox = self._find_board_region(frame)
        if bbox is not None:
            self._cached_bbox = bbox
            self._cache_counter = 0
            return self._crop(frame, bbox)

        # Fallback: use cached if fresh detection failed
        if self._cached_bbox is not None:
            self._cache_counter += 1
            return self._crop(frame, self._cached_bbox)

        return None

    def _find_board_region(self, frame):
        """
        Find the whiteboard using HSV color segmentation.

        Whiteboards are characterized by: high brightness (Value),
        low color saturation. The red patterned wall behind has
        high saturation, making it easy to separate.

        Returns
        -------
        tuple or None
            (x, y, w, h) bounding box of the whiteboard, or None.
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Whiteboard mask: low saturation + high value (bright, non-colorful)
        # Tuned for white/cream whiteboard against colored wall
        white_mask = cv2.inRange(hsv, (0, 0, 140), (180, 80, 255))

        # Morphological cleanup: close small gaps, remove noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        # Close gaps (fill holes in the whiteboard mask)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close)
        # Remove small noise blobs
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_open)

        # Find contours of white regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest white region (should be the whiteboard)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Must be at least BOARD_MIN_AREA_RATIO of frame
        if area < frame_area * config.BOARD_MIN_AREA_RATIO:
            return None

        # Get bounding rectangle
        x, y, bw, bh = cv2.boundingRect(largest)

        # Add small padding (5px) but stay within frame
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2 * pad)
        bh = min(h - y, bh + 2 * pad)

        # Sanity check: board must be reasonably sized
        if bw < 100 or bh < 100:
            return None

        return (x, y, bw, bh)

    def _crop(self, frame, bbox):
        """Crop the frame using the bounding box."""
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w].copy()

    def get_board_bbox(self):
        """Get the cached board bounding box (x, y, w, h) or None."""
        return self._cached_bbox
