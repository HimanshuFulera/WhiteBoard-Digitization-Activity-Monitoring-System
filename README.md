# Whiteboard Notes Extraction & Instructor Analytics

An intelligent computer vision pipeline that automatically extracts, compiles, and formats handwritten notes from lecture videos while simultaneously generating analytics on instructor behavior and engagement.

## Features

- **Automated Note Digitization:** Detects and crops whiteboards, extracting handwritten text using PaddleOCR.
- **Smart Occlusion Handling:** Uses YOLOv8 Pose estimation to track the instructor, ensuring text is only extracted when the board is clearly visible.
- **AI-Powered Refinement:** Processes raw OCR output through Gemini 2.5 Flash to fix typos, remove duplicates, and format cohesive lecture notes.
- **Instructor Analytics:** Tracks instructor positioning and facing direction to generate insights like active teaching time vs. idle time.
- **Efficient Processing:** Utilizes frame sampling and SSIM change detection to skip redundant frames and optimize processing speed.

## Technologies Used

- **Ultralytics YOLOv8 (Pose):** For detecting 17 body keypoints, tracking instructor movement, and calculating board occlusion.
- **PaddleOCR:** For high-quality handwriting recognition and text extraction.
- **Google Gemini 2.5 Flash:** For NLP-based text cleanup and formatting.
- **OpenCV & Scikit-Image:** For computer vision tasks including color thresholding, contrast enhancement, and structural similarity (SSIM) checks.
- **Python:** The core programming language powering the logic.

## Project Workflow

1. **Video Ingestion & Sampling:** The system reads lecture videos and samples frames to optimize processing time.
2. **Whiteboard Detection:** Isolates the whiteboard area using color filtering.
3. **Instructor Tracking:** Analyzes instructor pose and skips frames where they block the board (>30% occlusion).
4. **Change Detection:** Compares the current unoccluded board to the previous one; skips OCR if no new ink is detected.
5. **Text Extraction:** Enhances the image and runs PaddleOCR to read the text.
6. **Note Compilation:** Stitches overlapping text as the instructor writes incrementally.
7. **Analytics Generation:** Outputs a JSON file with instructor behavioral metrics upon completion.
8. **AI Refinement:** Sends raw text to Gemini for final cleanup, producing perfectly formatted lecture notes.

## Getting Started

1. Ensure you have Python installed.
2. Double-click **`run.bat`** (Windows) to automatically set up the virtual environment (`.venv`), install all dependencies, and begin execution via `main.py`.
