# Whiteboard Notes Extraction & Instructor Analytics
## Project Architecture & Workflow

This document provides a crisp overview of the system's architecture, the core technologies used, and the exact step-by-step workflow that occurs when the project is executed.

---

## 1. Technologies Used

*   **Ultralytics YOLOv8 (Pose):** A deep learning model used to detect the instructor's 17 body keypoints, allowing the system to track their movement, board interaction, and facing direction (board vs. students).
*   **PaddleOCR:** A state-of-the-art Optical Character Recognition engine optimized for extracting handwritten text and complex layouts from images.
*   **Google Gemini 2.5 Flash:** A Large Language Model used as a post-processor to clean up raw OCR data, fix typos, remove duplicates, and format the final text.
*   **OpenCV & Scikit-Image:** Used for all computer vision operations (color thresholding, contrast enhancement, SSIM change detection).

---

## 2. File Structure & Responsibilities

### Core Execution
*   **`run.bat`**: The entry point. It creates the Python virtual environment (`.venv`), installs dependencies, and runs `main.py`.
*   **`main.py`**: The central brain. It loops through the video frames and passes data sequentially between all the modules below.
*   **`config.py`**: The global settings file storing model paths, OCR thresholds, and output directories.

### Processing Modules (`modules/`)
*   **`frame_sampler.py`**: Reads the video and skips frames (e.g., extracting 2 FPS) to speed up processing.
*   **`whiteboard_detector.py`**: Uses color filtering to find and crop only the whiteboard from the background.
*   **`instructor_tracking.py`**: Runs the YOLOv8 Pose model to find the teacher, track their face to see where they are looking, and calculate if their body is blocking the board.
*   **`change_detector.py`**: Compares the current board to the previous board using SSIM to see if the teacher wrote new ink.
*   **`image_enhancer.py`**: Sharpens and enhances the contrast of the whiteboard crop so the OCR can read it easily.
*   **`ocr_engine.py`**: Executes PaddleOCR to read the text off the enhanced image.
*   **`note_compiler.py`**: Stitches overlapping sentences together as the teacher writes incrementally.
*   **`ai_refiner.py`**: Sends the final raw text to the Gemini API for cleanup and formatting.
*   **`analytics.py`**: Calculates the final tracking percentages (e.g., active teaching time) and saves them to a JSON file.

---

## 3. The Execution Workflow

When a user double-clicks **`run.bat`**, the following workflow is triggered in `main.py`:

1.  **Video Ingestion:** `main.py` calls `frame_sampler.py` to pull a frame from the video twice every second.
2.  **Whiteboard Cropping:** The frame is passed to `whiteboard_detector.py` to crop out the wall and isolate the board.
3.  **Instructor Tracking:** The frame is passed to `instructor_tracking.py`. 
    *   *Gating Check:* If the instructor is blocking >30% of the board, `main.py` instantly skips to the next frame to avoid reading garbage text.
4.  **Change Detection:** If the board is clear, the crop goes to `change_detector.py`. 
    *   *Gating Check:* If the ink hasn't changed since the last frame, `main.py` skips the OCR step to save time.
5.  **Text Extraction:** If new ink is found, `image_enhancer.py` cleans the image, and `ocr_engine.py` extracts the text.
6.  **Compilation:** The extracted text is handed to `note_compiler.py` to be stitched into the running document.
7.  **(Loop repeats until the video ends)**
8.  **Analytics Generation:** Once the video is done, `main.py` calls `analytics.py` to generate `analytics.json`, detailing how long the instructor spent facing the students vs. writing.
9.  **AI Refinement:** Finally, `main.py` saves the raw notes and triggers `ai_refiner.py`, which uses Gemini to output the perfectly formatted `refined.txt` lecture notes.
