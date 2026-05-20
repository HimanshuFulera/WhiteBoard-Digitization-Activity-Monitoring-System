"""
Image Enhancer Module.

Prepares cropped whiteboard images for OCR by applying:
  - Resolution upscaling
  - Contrast enhancement (CLAHE)
  - Denoising
  - Sharpening

IMPORTANT: Does NOT binarize the image. PaddleOCR works significantly
better on enhanced grayscale/color images than on binary thresholded ones.
"""

import cv2
import numpy as np
import config


def enhance_for_ocr(board_crop):
    """
    Enhance a whiteboard crop for optimal OCR accuracy.

    The pipeline:
    1. Upscale for better character resolution
    2. Convert to grayscale
    3. Denoise to remove video compression artifacts
    4. CLAHE for local contrast enhancement
    5. Sharpen to make text edges crisp

    Parameters
    ----------
    board_crop : np.ndarray
        BGR cropped whiteboard image.

    Returns
    -------
    np.ndarray
        Enhanced grayscale image ready for OCR.
    """
    # Step 1: Upscale for better character resolution
    h, w = board_crop.shape[:2]
    factor = config.UPSCALE_FACTOR
    upscaled = cv2.resize(board_crop, (w * factor, h * factor),
                          interpolation=cv2.INTER_CUBIC)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Step 3: Denoise — removes video compression artifacts
    # h=10 for moderate denoising without losing text detail
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                         searchWindowSize=21)

    # Step 4: CLAHE — adaptive contrast enhancement
    # Makes faint text more visible without blowing out bright areas
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                            tileGridSize=config.CLAHE_GRID_SIZE)
    enhanced = clahe.apply(denoised)

    # Step 5: Sharpen — makes text edges crisp
    sharpen_kernel = np.array([
        [0,  -1,  0],
        [-1,  5, -1],
        [0,  -1,  0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    return sharpened


def enhance_for_ssim(board_crop):
    """
    Lightweight preprocessing for SSIM comparison only.
    Just grayscale + light denoise — fast, not for OCR.

    Parameters
    ----------
    board_crop : np.ndarray
        BGR cropped whiteboard image.

    Returns
    -------
    np.ndarray
        Grayscale image for SSIM comparison.
    """
    gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)
