"""
Analytics Module.

Computes instructor activity analytics and exports to JSON.
"""

import json
import os
import config

def compute_analytics(tracking_data, fps, total_frames, frame_skip):
    """
    Compute instructor activity analytics based on pose tracking.
    """
    total_duration = total_frames / fps if fps > 0 else 0
    time_per_processed_frame = frame_skip / fps if fps > 0 else 0

    visible_frames = tracking_data.get("frames_visible", 0)
    processed_frames = tracking_data.get("frames_total", 0)
    interacting_frames = tracking_data.get("frames_interacting", 0)
    facing_students_frames = tracking_data.get("frames_facing_students", 0)
    facing_board_frames = tracking_data.get("frames_facing_board", 0)
    idle_frames = tracking_data.get("frames_idle", 0)

    visible_duration = visible_frames * time_per_processed_frame
    out_of_frame_duration = max(0, total_duration - visible_duration)
    
    facing_students_sec = facing_students_frames * time_per_processed_frame
    facing_board_sec = facing_board_frames * time_per_processed_frame
    idle_time_sec = idle_frames * time_per_processed_frame
    
    # Active teaching is when the instructor is either interacting with board or facing students
    active_teaching_frames = interacting_frames + facing_students_frames
    active_teaching_sec = active_teaching_frames * time_per_processed_frame
    
    # Cap active teaching to total visible time just in case of slight overlap in frame definitions
    active_teaching_sec = min(active_teaching_sec, visible_duration)

    analytics = {
        "total_session_duration_sec": round(total_duration, 2),
        "out_of_frame_duration_sec": round(out_of_frame_duration, 2),
        "facing_students_sec": round(facing_students_sec, 2),
        "facing_board_sec": round(facing_board_sec, 2),
        "active_teaching_sec": round(active_teaching_sec, 2),
        "idle_time_sec": round(idle_time_sec, 2),
        "board_interaction_duration_sec": round(interacting_frames * time_per_processed_frame, 2),
        "total_processed_frames": processed_frames,
    }

    return analytics

def save_analytics(analytics, output_dir):
    """Save analytics to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "analytics.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)
    print(f"[Analytics] JSON saved to {json_path}")

def print_analytics_summary(analytics):
    """Print a formatted summary of analytics to console."""
    print("\n" + "=" * 60)
    print("  INSTRUCTOR ACTIVITY ANALYTICS SUMMARY")
    print("=" * 60)
    print(f"  Total Session Duration:      {analytics['total_session_duration_sec']} sec")
    print(f"  Out of Frame:                {analytics['out_of_frame_duration_sec']} sec")
    print(f"  Facing Students (Speaking):  {analytics['facing_students_sec']} sec")
    print(f"  Facing Board (Writing):      {analytics['facing_board_sec']} sec")
    print(f"  Active Teaching Time:        {analytics['active_teaching_sec']} sec")
    print(f"  Idle Time:                   {analytics['idle_time_sec']} sec")
    print("=" * 60 + "\n")
