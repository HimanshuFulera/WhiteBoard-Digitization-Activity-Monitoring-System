"""
Frame Sampler Module.

Reads video frames at configurable intervals and filters out
blurry frames using Laplacian variance analysis.
"""

import cv2
import numpy as np
import config


def get_video_info(video_path):
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def compute_blur_score(frame):
    """
    Compute blur score using Laplacian variance.
    Higher = sharper.  Lower = blurrier.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sample_frames(video_path, frame_skip=None):
    """
    Generator that yields (frame_index, timestamp_sec, frame) tuples.

    Skips frames based on FRAME_SKIP and filters out blurry frames
    using Laplacian variance.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_skip : int, optional
        Override config.FRAME_SKIP. Process every Nth frame.

    Yields
    ------
    tuple of (int, float, np.ndarray)
        (frame_index, timestamp_in_seconds, BGR_frame)
    """
    if frame_skip is None:
        frame_skip = config.FRAME_SKIP

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Check blur
            blur_score = compute_blur_score(frame)
            if blur_score >= config.BLUR_THRESHOLD:
                timestamp = frame_idx / fps if fps > 0 else 0
                yield frame_idx, timestamp, frame

        frame_idx += 1

    cap.release()
