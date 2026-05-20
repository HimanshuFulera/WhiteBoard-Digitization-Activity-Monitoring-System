"""
Configuration file for Whiteboard Notes Extraction System.

All tunable parameters are centralized here.
"""

import os

# ========================== PATHS ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

VIDEO_PATH = os.path.join(INPUT_DIR, "test_video.mp4")

NOTES_DIR = os.path.join(OUTPUT_DIR, "extracted_notes")
SNAPSHOTS_DIR = os.path.join(OUTPUT_DIR, "board_snapshots")

NOTES_TXT = os.path.join(NOTES_DIR, "lecture_notes.txt")
NOTES_PDF = os.path.join(NOTES_DIR, "lecture_notes.pdf")

# ========================== VIDEO ================================
FRAME_SKIP = 15           # Process every Nth frame (30fps video → 2 fps effective)

# ========================== BLUR DETECTION =======================
BLUR_THRESHOLD = 50.0     # Laplacian variance below this = blurry frame (skip)

# ========================== WHITEBOARD DETECTION =================
BOARD_MIN_AREA_RATIO = 0.05   # Board must be at least 5% of frame area
BOARD_MAX_AREA_RATIO = 0.95   # Board must be at most 95% of frame area
BOARD_APPROX_EPSILON = 0.02   # Contour approximation factor
BOARD_CACHE_FRAMES = 30       # Re-detect board every N processed frames
BOARD_BRIGHTNESS_MIN = 120    # Minimum mean brightness for a whiteboard region

# ========================== CHANGE DETECTION =====================
SSIM_THRESHOLD = 0.88         # Board content change sensitivity
                              # Lower = more sensitive (more OCR calls)
                              # Higher = less sensitive (fewer OCR calls)
COOLDOWN_FRAMES = 30          # Minimum processed frames between OCR calls

# ========================== IMAGE ENHANCEMENT ====================
UPSCALE_FACTOR = 2            # Upscale board crop before OCR (2x resolution)
CLAHE_CLIP_LIMIT = 3.0        # Contrast enhancement strength
CLAHE_GRID_SIZE = (8, 8)      # Contrast enhancement grid

# ========================== OCR (PaddleOCR) ======================
OCR_LANG = "en"
OCR_USE_ANGLE_CLS = True      # Enable text direction classification
OCR_DET_DB_THRESH = 0.3       # Text detection threshold
OCR_DET_DB_BOX_THRESH = 0.5   # Text box detection threshold
OCR_REC_THRESH = 0.5          # Recognition confidence threshold (reject below)
OCR_MIN_TEXT_LENGTH = 3        # Minimum characters for valid text
OCR_MIN_WORDS = 1              # Minimum words for valid text

# ========================== NOTE COMPILATION =====================
SIMILARITY_THRESHOLD = 0.45   # Notes with similarity > this are considered duplicates
MIN_NEW_CONTENT_RATIO = 0.15  # New OCR must have >15% new content to be kept

# ========================== INSTRUCTOR TRACKING ==================
YOLO_MODEL = "yolov8n-pose.pt"     # Lightweight YOLOv8 pose model for tracking and facing direction
YOLO_CONFIDENCE = 0.5         # Minimum confidence to accept a detection
BOARD_INTERACTION_THRESHOLD = 0.10  # Minimum occlusion ratio to count as interacting

# ========================== ANALYTICS ============================
ANALYTICS_DIR = os.path.join(OUTPUT_DIR, "analytics")
ANALYTICS_JSON = os.path.join(ANALYTICS_DIR, "analytics.json")

# Ensure output directories exist
os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)
