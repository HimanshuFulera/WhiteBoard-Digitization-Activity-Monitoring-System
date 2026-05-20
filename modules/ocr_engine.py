"""
OCR Engine Module.

Wraps PaddleOCR for text extraction from enhanced whiteboard images.
Handles initialization, text extraction, confidence filtering,
and spatial ordering of detected text blocks.
"""

import re
import numpy as np
from paddleocr import PaddleOCR
import config


# Lazy-initialized singleton — PaddleOCR is heavy to load
_paddle_reader = None


def _get_reader():
    """Get or create the PaddleOCR reader (singleton)."""
    global _paddle_reader
    if _paddle_reader is None:
        print("[OCR] Initializing PaddleOCR engine...")
        _paddle_reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang=config.OCR_LANG,
            text_det_thresh=config.OCR_DET_DB_THRESH,
            text_det_box_thresh=config.OCR_DET_DB_BOX_THRESH,
            text_rec_score_thresh=config.OCR_REC_THRESH,
            ocr_version="PP-OCRv4",
            enable_mkldnn=False,
        )
        print("[OCR] PaddleOCR ready.")
    return _paddle_reader


def clean_text(text):
    """
    Clean raw OCR text:
    - Strip whitespace
    - Remove isolated special characters
    - Collapse multiple spaces
    """
    text = text.strip()

    # Remove lines that are just punctuation/symbols
    if re.match(r'^[\W_]+$', text):
        return ""

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text


def is_valid_text(text):
    """
    Check if extracted text meets minimum quality standards.
    """
    if len(text) < config.OCR_MIN_TEXT_LENGTH:
        return False

    words = text.split()
    if len(words) < config.OCR_MIN_WORDS:
        return False

    # Must contain at least some alphabetic characters
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return False

    return True


def extract_text(enhanced_image):
    """
    Extract text from an enhanced whiteboard image using PaddleOCR.

    Text blocks are sorted top-to-bottom, left-to-right to preserve
    the natural reading order of whiteboard content.

    Parameters
    ----------
    enhanced_image : np.ndarray
        Enhanced grayscale or BGR whiteboard image.

    Returns
    -------
    list of str
        List of text lines extracted, in reading order.
        Empty list if no valid text found.
    """
    reader = _get_reader()

    # PaddleOCR expects uint8
    if enhanced_image.dtype != np.uint8:
        enhanced_image = enhanced_image.astype(np.uint8)

    # If grayscale, convert to 3-channel for PaddleOCR
    if len(enhanced_image.shape) == 2:
        enhanced_image = np.stack([enhanced_image] * 3, axis=-1)

    # Run OCR
    result = reader.predict(enhanced_image)

    if not result or len(result) == 0:
        return []

    # Extract and filter text blocks with their positions
    text_blocks = []

    for item in result:
        rec_texts = item.get("rec_texts", [])
        rec_scores = item.get("rec_scores", [])
        dt_polys = item.get("dt_polys", [])

        for i, text in enumerate(rec_texts):
            confidence = rec_scores[i] if i < len(rec_scores) else 0.0
            poly = dt_polys[i] if i < len(dt_polys) else None

            # Filter by confidence
            if confidence < config.OCR_REC_THRESH:
                continue

            # Clean the text
            cleaned = clean_text(text)
            if not cleaned:
                continue

            # Get vertical/horizontal center for sorting
            if poly is not None and len(poly) >= 4:
                y_center = np.mean([pt[1] for pt in poly])
                x_center = np.mean([pt[0] for pt in poly])
            else:
                y_center = 0
                x_center = 0

            text_blocks.append({
                "text": cleaned,
                "confidence": confidence,
                "y": y_center,
                "x": x_center,
            })

    if not text_blocks:
        return []

    # Sort by vertical position (top to bottom), then horizontal (left to right)
    # Group into rows: blocks within 30px vertical distance are on the same row
    text_blocks.sort(key=lambda b: b["y"])

    rows = []
    current_row = [text_blocks[0]]

    for block in text_blocks[1:]:
        # If this block is close vertically to the last one, same row
        if abs(block["y"] - current_row[-1]["y"]) < 30:
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]
    rows.append(current_row)

    # Sort each row left-to-right and join
    lines = []
    for row in rows:
        row.sort(key=lambda b: b["x"])
        line_text = " ".join(b["text"] for b in row)
        if is_valid_text(line_text):
            lines.append(line_text)

    return lines
