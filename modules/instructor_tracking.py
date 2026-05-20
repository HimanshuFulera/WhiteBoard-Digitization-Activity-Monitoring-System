"""
Instructor Tracking Module.

Uses YOLOv8 to track the instructor's bounding box and computes how much
of the whiteboard they are occluding.
"""

from ultralytics import YOLO
import config

def compute_intersection_ratio(box_a, box_b):
    """
    Compute how much of box_b is occluded by box_a.
    Returns a float between 0.0 and 1.0.
    Boxes are in format (x1, y1, x2, y2).
    """
    if box_a is None or box_b is None:
        return 0.0

    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    # Calculate intersection area
    x_left = max(xa1, xb1)
    y_top = max(ya1, yb1)
    x_right = min(xa2, xb2)
    y_bottom = min(ya2, yb2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_b_area = (xb2 - xb1) * (yb2 - yb1)

    if box_b_area <= 0:
        return 0.0

    return intersection_area / box_b_area

class InstructorTracker:
    """Tracks the instructor across video frames using YOLOv8 Pose."""

    def __init__(self):
        print(f"[Tracker] Loading YOLOv8 model: {config.YOLO_MODEL}")
        self.model = YOLO(config.YOLO_MODEL)
        
        # Tracking metrics
        self.frames_total = 0           # Total frames processed
        self.frames_visible = 0         # Frames where instructor is detected
        self.frames_interacting = 0     # Frames where instructor occludes whiteboard
        
        # Advanced tracking
        self.frames_facing_students = 0
        self.frames_facing_board = 0
        self.frames_idle = 0
        
    def detect_instructor(self, frame):
        """
        Detect the instructor (person) and their pose in a frame.
        Returns a tuple: (best_bbox, is_facing_students)
        best_bbox is (x1, y1, x2, y2) or None.
        is_facing_students is a boolean or None.
        """
        results = self.model(frame, verbose=False, conf=config.YOLO_CONFIDENCE)
        
        best_bbox = None
        best_area = 0
        is_facing_students = False

        for result in results:
            boxes = result.boxes
            keypoints_obj = getattr(result, 'keypoints', None)
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                # Class 0 is 'person'
                if cls == 0 and conf >= config.YOLO_CONFIDENCE:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > best_area:
                        best_bbox = (float(x1), float(y1), float(x2), float(y2))
                        best_area = area
                        
                        # Check facial keypoints to determine facing direction
                        is_facing_students = False
                        if keypoints_obj is not None and keypoints_obj.conf is not None:
                            kpts_conf = keypoints_obj.conf[i].cpu().numpy()
                            # Keypoints 0 (nose), 1 (left eye), 2 (right eye)
                            if len(kpts_conf) >= 3:
                                face_conf = max(kpts_conf[0], kpts_conf[1], kpts_conf[2])
                                if face_conf > 0.5:
                                    is_facing_students = True

        return best_bbox, is_facing_students

    def compute_occlusion(self, instructor_bbox, board_bbox):
        """Returns the occlusion ratio of the whiteboard."""
        return compute_intersection_ratio(instructor_bbox, board_bbox)

    def update_tracking(self, instructor_bbox, is_facing_students, board_bbox=None):
        """
        Update the internal metrics with the current frame's detections.
        """
        self.frames_total += 1

        if instructor_bbox is not None:
            self.frames_visible += 1
            
            is_interacting = False
            if board_bbox is not None:
                occlusion = self.compute_occlusion(instructor_bbox, board_bbox)
                if occlusion > config.BOARD_INTERACTION_THRESHOLD:
                    self.frames_interacting += 1
                    is_interacting = True
                    
            if is_facing_students:
                self.frames_facing_students += 1
            else:
                self.frames_facing_board += 1
                
            # Idle: not facing students AND not interacting with the board
            if not is_facing_students and not is_interacting:
                self.frames_idle += 1

    def get_tracking_data(self):
        """Get the aggregated tracking metrics."""
        return {
            "frames_total": self.frames_total,
            "frames_visible": self.frames_visible,
            "frames_interacting": self.frames_interacting,
            "frames_facing_students": self.frames_facing_students,
            "frames_facing_board": self.frames_facing_board,
            "frames_idle": self.frames_idle
        }
