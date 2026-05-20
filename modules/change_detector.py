"""
Change Detector Module.

Uses SSIM (Structural Similarity Index) to compare consecutive
whiteboard crops and determine if the content has changed enough
to warrant a new OCR extraction.

Also includes a "board clarity" check that rejects frames where
a person (teacher) is blocking a significant portion of the board.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import config


def compute_board_clarity(board_crop):
    """
    Compute how "clear" the board is — i.e., what fraction is
    actually white/bright (unblocked by the teacher).

    A high clarity (>0.5) means most of the board is visible.
    A low clarity (<0.3) means the teacher is blocking much of it.

    Parameters
    ----------
    board_crop : np.ndarray
        BGR cropped whiteboard image.

    Returns
    -------
    float
        Ratio of bright pixels (0.0 to 1.0).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)

    # Bright pixels = whiteboard surface; dark pixels = teacher/shadow
    _, bright_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    bright_ratio = np.count_nonzero(bright_mask) / bright_mask.size
    return bright_ratio


class ChangeDetector:
    """
    Detects meaningful changes in whiteboard content between frames.

    Uses SSIM comparison + cooldown timer + clarity check to ensure:
    1. Content has actually changed (SSIM)
    2. Enough time has passed since last OCR (cooldown)
    3. The board is mostly visible / teacher not blocking (clarity)
    """

    # Minimum fraction of board that must be bright (not blocked by teacher)
    CLARITY_THRESHOLD = 0.45

    def __init__(self):
        self._prev_gray = None
        self._frames_since_ocr = config.COOLDOWN_FRAMES  # Allow first OCR immediately
        self._best_crop = None        # Best board crop seen in current interval
        self._best_clarity = 0.0      # Clarity score of the best crop

    def has_changed(self, board_crop):
        """
        Check if the board content has changed enough since last OCR,
        AND the board is sufficiently clear (teacher not blocking).

        Parameters
        ----------
        board_crop : np.ndarray
            BGR cropped whiteboard image.

        Returns
        -------
        bool
            True if content changed, cooldown elapsed, AND board is clear.
        """
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (400, 300))  # Normalize size for consistent SSIM

        self._frames_since_ocr += 1

        # Compute clarity
        clarity = compute_board_clarity(board_crop)

        # Track the clearest crop during this interval
        if clarity > self._best_clarity:
            self._best_clarity = clarity
            self._best_crop = board_crop.copy()

        # First frame — trigger OCR if board is clear enough
        if self._prev_gray is None:
            if clarity >= self.CLARITY_THRESHOLD:
                self._prev_gray = gray
                self._frames_since_ocr = 0
                self._best_clarity = 0.0
                return True
            return False

        # Check cooldown
        if self._frames_since_ocr < config.COOLDOWN_FRAMES:
            return False

        # Compute SSIM
        score = ssim(self._prev_gray, gray)

        # Content changed AND board is clear enough
        if score < config.SSIM_THRESHOLD and clarity >= self.CLARITY_THRESHOLD:
            self._prev_gray = gray
            self._frames_since_ocr = 0
            self._best_clarity = 0.0
            return True

        return False

    def get_best_crop(self):
        """Get the clearest board crop seen in the current interval."""
        return self._best_crop

    def force_update(self, board_crop):
        """Force update the reference frame."""
        gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)
        self._prev_gray = cv2.resize(gray, (400, 300))
        self._frames_since_ocr = 0
        self._best_clarity = 0.0
