"""
Main Pipeline — Whiteboard Notes Extraction System.

Orchestrates the full processing pipeline:
1. Sample sharp frames from video
2. Detect and crop whiteboard
3. Detect content changes (SSIM)
4. Enhance image for OCR
5. Extract text with PaddleOCR
6. Compile into clean lecture notes

Usage:
    python main.py
"""

import os
import sys
import time
import cv2

import config
from modules.frame_sampler import get_video_info, sample_frames
from modules.whiteboard_detector import WhiteboardDetector
from modules.change_detector import ChangeDetector
from modules.image_enhancer import enhance_for_ocr
from modules.ocr_engine import extract_text
from modules.note_compiler import NoteCompiler
from modules.instructor_tracking import InstructorTracker
from modules.analytics import compute_analytics, save_analytics, print_analytics_summary


def run_pipeline():
    """Run the full notes extraction pipeline."""
    print("=" * 60)
    print("  WHITEBOARD NOTES EXTRACTION SYSTEM")
    print("=" * 60)

    # ── Step 0: Validate input ─────────────────────────────────
    if not os.path.exists(config.VIDEO_PATH):
        print(f"\n[ERROR] Video not found: {config.VIDEO_PATH}")
        print("Place your video file at the path above and retry.")
        sys.exit(1)

    video_info = get_video_info(config.VIDEO_PATH)
    print(f"\n[Video] {os.path.basename(config.VIDEO_PATH)}")
    print(f"  Resolution : {video_info['width']}x{video_info['height']}")
    print(f"  FPS        : {video_info['fps']:.1f}")
    print(f"  Frames     : {video_info['total_frames']}")
    print(f"  Duration   : {video_info['duration_sec']:.1f} sec")
    print(f"  Frame skip : {config.FRAME_SKIP} (processing every {config.FRAME_SKIP}th frame)")

    total_to_process = video_info['total_frames'] // config.FRAME_SKIP
    print(f"  Frames to process: ~{total_to_process}")

    # ── Step 1: Initialize modules ─────────────────────────────
    print("\n[Init] Loading modules...")
    detector = WhiteboardDetector()
    change_detector = ChangeDetector()
    compiler = NoteCompiler()
    tracker = InstructorTracker()

    # Counters
    frames_processed = 0
    frames_with_board = 0
    ocr_runs = 0
    start_time = time.time()

    # ── Step 2: Process video frames ───────────────────────────
    print("\n[Processing] Starting video analysis...\n")

    for frame_idx, timestamp, frame in sample_frames(config.VIDEO_PATH):
        frames_processed += 1

        # Progress indicator
        progress = (frame_idx / video_info['total_frames']) * 100
        sys.stdout.write(
            f"\r  [{progress:5.1f}%] Frame {frame_idx}/{video_info['total_frames']}"
            f" | Board crops: {frames_with_board}"
            f" | OCR runs: {ocr_runs}"
        )
        sys.stdout.flush()

        # ── Detect whiteboard ──
        board_crop = detector.detect(frame)
        board_xywh = detector.get_board_bbox()
        
        # Convert board bbox to (x1, y1, x2, y2)
        board_bbox = None
        if board_xywh is not None:
            x, y, w, h = board_xywh
            board_bbox = (x, y, x + w, y + h)

        # ── Detect instructor ──
        instructor_bbox, is_facing_students = tracker.detect_instructor(frame)
        tracker.update_tracking(instructor_bbox, is_facing_students, board_bbox)

        if board_crop is None:
            continue

        frames_with_board += 1

        # ── Check if content changed ──
        if not change_detector.has_changed(board_crop):
            continue
            
        # ── OCR Occlusion Gate ──
        # Don't run OCR if teacher is actively blocking the board
        occlusion = tracker.compute_occlusion(instructor_bbox, board_bbox)
        if occlusion > 0.30:
            continue

        # ── Content changed! Enhance and run OCR ──
        enhanced = enhance_for_ocr(board_crop)

        # Save snapshot for debugging
        snap_path = os.path.join(config.SNAPSHOTS_DIR, f"board_{ocr_runs:03d}.png")
        cv2.imwrite(snap_path, enhanced)

        # Extract text
        lines = extract_text(enhanced)
        ocr_runs += 1

        if lines:
            print(f"\n  [OCR #{ocr_runs}] at {timestamp:.1f}s — {len(lines)} lines extracted")
            for line in lines:
                print(f"    > {line}")

            # Add to note compiler
            compiler.add_extraction(lines)
        else:
            print(f"\n  [OCR #{ocr_runs}] at {timestamp:.1f}s — no valid text found")

    # ── Step 3: Finalize ───────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n\n{'=' * 60}")
    print(f"  PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Time elapsed    : {elapsed:.1f} sec")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Board detections: {frames_with_board}")
    print(f"  OCR runs        : {ocr_runs}")
    print(f"  Note sections   : {compiler.get_section_count()}")

    # ── Step 3: Compute and save analytics ─────────────────────────
    tracking_data = tracker.get_tracking_data()
    analytics = compute_analytics(tracking_data, video_info['fps'], video_info['total_frames'], config.FRAME_SKIP)
    save_analytics(analytics, config.ANALYTICS_DIR)
    print_analytics_summary(analytics)

    # ── Step 3: Save notes ─────────────────────────────────────────
    # First save the raw OCR output
    raw_notes_path = os.path.join(config.NOTES_DIR, "raw_extracted_notes.txt")
    with open(raw_notes_path, "w", encoding="utf-8") as f:
        f.write(compiler.get_notes())
    print(f"\n[Saving] Raw notes saved to: {raw_notes_path}")
    
    # Then refine it using Gemini API
    refined_notes_path = config.NOTES_TXT
    
    # Try to import and use the AI refiner
    try:
        from modules.ai_refiner import refine_notes_with_gemini
        print("\n[AI Refiner] Starting notes refinement...")
        success = refine_notes_with_gemini(raw_notes_path, refined_notes_path)
        if success:
             print("[AI Refiner] Refinement complete.")
        else:
             print("[AI Refiner] Refinement failed, using raw notes as final.")
             import shutil
             shutil.copyfile(raw_notes_path, refined_notes_path)
    except ImportError:
        print("[AI Refiner] ai_refiner module not found. Skipping refinement.")
        import shutil
        shutil.copyfile(raw_notes_path, refined_notes_path)

    # ── Step 4: Save notes ─────────────────────────────────────
    print(f"\n[Saving] Generating lecture notes PDF...")
    # NOTE: compiler.save_txt() is skipped here because Step 3 handled it via AI Refiner
    pdf_path = compiler.save_pdf()

    # Print final notes (from file)
    print(f"\n{'=' * 60}")
    if os.path.exists(config.NOTES_TXT):
        with open(config.NOTES_TXT, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print(compiler.get_notes())

    print(f"\n[Done] Files saved:")
    print(f"  Raw Notes   : {raw_notes_path}")
    print(f"  Final Notes : {config.NOTES_TXT}")
    print(f"  Notes (PDF) : {pdf_path}")
    print(f"  Snapshots   : {config.SNAPSHOTS_DIR}")


if __name__ == "__main__":
    run_pipeline()
