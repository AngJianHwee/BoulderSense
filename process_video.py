!pip install mediapipe opencv-python 
!pip show mediapipe opencv-python
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import os
import logging
import json
import time
import sys

# Clear any pre-existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "output_video.mp4"
OUTPUT_VIDEO_TRAIL = "output_video_com_trail.mp4"
OUTPUT_FRAMES_DIR = "output_frames"
OUTPUT_JSON = "pose_annotations.json"
MODEL_PATH = "pose_landmarker.task"

# COM Smoothing settings (EMA - Exponential Moving Average)
EMA_ALPHA = 0.3  # Lower = more smoothing, higher = more responsive (0.1-0.5 typical)

# COM Trail settings
TRAIL_MAX_ALPHA = 0.8
TRAIL_MIN_ALPHA = 0.1

# Create output directory
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
logger.info("Step 0 complete: output directory ready: %s", OUTPUT_FRAMES_DIR)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    import urllib.request
    logger.info("Downloading model to %s...", MODEL_PATH)
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    logger.info("Download complete.")
else:
    logger.info("Model already exists: %s", MODEL_PATH)

logger.info("Model setup complete.")


# ============================================================
# STEP 1: Create PoseLandmarker object
# ============================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)
logger.info("Step 1 complete: PoseLandmarker created.")


# ============================================================
# STEP 2: Open input video
# ============================================================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    logger.error("Could not open video %s", INPUT_VIDEO)
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

logger.info("Video info: %sx%s, %s FPS, %s frames", width, height, fps, total_frames)
logger.info("Step 2 complete: input video opened.")


# ============================================================
# STEP 3: Setup video writers for output
# ============================================================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    logger.error("Could not open output video %s", OUTPUT_VIDEO)
    cap.release()
    detector.close()
    exit(1)

# Second video writer for COM trail (full curve + pose overlay)
out_trail = cv2.VideoWriter(OUTPUT_VIDEO_TRAIL, fourcc, fps, (width, height))
if not out_trail.isOpened():
    logger.error("Could not open output trail video %s", OUTPUT_VIDEO_TRAIL)
    cap.release()
    out.release()
    detector.close()
    exit(1)
logger.info("Step 3 complete: output video writers ready.")


# ============================================================
# STEP 4: Define landmark indices for COM calculation
# ============================================================
# Landmark indices (MediaPipe Pose):
# 0: nose, 11: left_shoulder, 12: right_shoulder
# 23: left_hip, 24: right_hip
# 13: left_elbow, 14: right_elbow
# 15: left_wrist, 16: right_wrist
# 25: left_knee, 26: right_knee
# 27: left_ankle, 28: right_ankle
# 29: left_heel, 30: right_heel
# 31: left_foot_index, 32: right_foot_index

COM_LANDMARK_INDICES = {
    'head': [0],
    'torso': [11, 12, 23, 24],
    'left_arm': [13, 15],
    'right_arm': [14, 16],
    'left_leg': [25, 27],
    'right_leg': [26, 28],
}

SEGMENT_WEIGHTS = {
    'head': 0.07,
    'torso': 0.43,
    'left_arm': 0.05,
    'right_arm': 0.05,
    'left_leg': 0.16,
    'right_leg': 0.16,
}
logger.info("Step 4 complete: COM landmarks and segment weights configured.")


# ============================================================
# STEP 5: Function to calculate Center of Mass (raw)
# ============================================================
def calculate_com_raw(landmarks, image_width, image_height):
    """
    Calculate raw Center of Mass from pose landmarks (no smoothing).
    Returns (x, y) in pixel coordinates as floats.
    """
    com_x = 0.0
    com_y = 0.0
    total_weight = 0.0
    
    for segment_name, indices in COM_LANDMARK_INDICES.items():
        weight = SEGMENT_WEIGHTS[segment_name]
        segment_x = 0.0
        segment_y = 0.0
        valid_count = 0
        
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.5:
                    segment_x += lm.x * image_width
                    segment_y += lm.y * image_height
                    valid_count += 1
        
        if valid_count > 0:
            segment_x /= valid_count
            segment_y /= valid_count
            com_x += segment_x * weight
            com_y += segment_y * weight
            total_weight += weight
    
    if total_weight > 0:
        com_x /= total_weight
        com_y /= total_weight
    
    return com_x, com_y


# ============================================================
# STEP 6: EMA Smoothing for COM
# ============================================================
class EMASmoother:
    """Exponential Moving Average smoother for 2D points."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0
        self.initialized = False
    
    def update(self, x, y):
        if not self.initialized:
            self.smoothed_x = x
            self.smoothed_y = y
            self.initialized = True
        else:
            self.smoothed_x = self.alpha * x + (1 - self.alpha) * self.smoothed_x
            self.smoothed_y = self.alpha * y + (1 - self.alpha) * self.smoothed_y
        return self.smoothed_x, self.smoothed_y
    
    def reset(self):
        self.initialized = False
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0


ema_smoother = EMASmoother(alpha=EMA_ALPHA)


# ============================================================
# STEP 7: Function to draw landmarks and COM on frame
# ============================================================
def draw_landmarks_and_com(frame, detection_result, com_point, com_raw=None):
    """
    Draw pose landmarks and COM on the frame.
    Shows both raw (small) and smoothed (large) COM if both provided.
    Uses modern MediaPipe Tasks drawing utilities.
    """
    annotated_frame = frame.copy()  # BGR

    pose_landmarks_list = detection_result.pose_landmarks
    if pose_landmarks_list:
        # drawing_utils expects RGB
        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=rgb,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style,
            )

        annotated_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Draw raw COM as small orange dot (if available)
    if com_raw is not None:
        rx, ry = int(com_raw[0]), int(com_raw[1])
        cv2.circle(annotated_frame, (rx, ry), 5, (0, 165, 255), -1)  # Orange
        cv2.circle(annotated_frame, (rx, ry), 7, (255, 255, 255), 1)

    # Draw smoothed COM as red circle with crosshair
    if com_point is not None:
        com_x, com_y = int(com_point[0]), int(com_point[1])
        # Red circle
        cv2.circle(annotated_frame, (com_x, com_y), 10, (0, 0, 255), -1)
        # White border
        cv2.circle(annotated_frame, (com_x, com_y), 12, (255, 255, 255), 2)
        # Crosshair
        cv2.line(annotated_frame, (com_x - 20, com_y), (com_x + 20, com_y), (0, 0, 255), 2)
        cv2.line(annotated_frame, (com_x, com_y - 20), (com_x, com_y + 20), (0, 0, 255), 2)
        # Label
        cv2.putText(annotated_frame, "COM (smoothed)", (com_x + 15, com_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return annotated_frame


# ============================================================
# STEP 8: Function to draw full COM trail curve + pose overlay
# ============================================================
def draw_full_trail_with_pose(frame, detection_result, all_com_history, current_frame_idx):
    """
    Draw the full COM trajectory curve across all frames + current pose overlay.
    This creates the 'trail video' showing stability of COM path.
    """
    trail_frame = frame.copy()

    # Draw pose landmarks (current frame pose)
    pose_landmarks_list = detection_result.pose_landmarks
    if pose_landmarks_list:
        rgb = cv2.cvtColor(trail_frame, cv2.COLOR_BGR2RGB)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=rgb,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style,
            )

        trail_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Draw full COM trajectory curve
    num_points = len(all_com_history)
    if num_points >= 2:
        # Draw all segments with gradient color (blue -> green -> red)
        for i in range(1, num_points):
            alpha_ratio = i / num_points

            pt1 = (int(all_com_history[i-1][0]), int(all_com_history[i-1][1]))
            pt2 = (int(all_com_history[i][0]), int(all_com_history[i][1]))

            # Color gradient: blue (start) -> cyan -> green -> yellow -> red (end)
            if alpha_ratio < 0.25:
                # Blue to cyan
                t = alpha_ratio * 4
                color = (int(255 * (1 - t)), int(255 * t), 255)
            elif alpha_ratio < 0.5:
                # Cyan to green
                t = (alpha_ratio - 0.25) * 4
                color = (0, 255, int(255 * (1 - t)))
            elif alpha_ratio < 0.75:
                # Green to yellow
                t = (alpha_ratio - 0.5) * 4
                color = (0, 255, int(255 * t))
            else:
                # Yellow to red
                t = (alpha_ratio - 0.75) * 4
                color = (0, int(255 * (1 - t)), 255)

            # Alpha increases over time (older = more transparent)
            alpha = TRAIL_MIN_ALPHA + (TRAIL_MAX_ALPHA - TRAIL_MIN_ALPHA) * alpha_ratio
            thickness = max(1, int(3 * alpha))

            overlay = trail_frame.copy()
            cv2.line(overlay, pt1, pt2, color, thickness)
            cv2.addWeighted(overlay, alpha, trail_frame, 1 - alpha, 0, trail_frame)

        # Mark start point (green circle)
        start_pt = (int(all_com_history[0][0]), int(all_com_history[0][1]))
        cv2.circle(trail_frame, start_pt, 8, (0, 255, 0), -1)
        cv2.circle(trail_frame, start_pt, 10, (255, 255, 255), 2)
        cv2.putText(trail_frame, "START", (start_pt[0] + 12, start_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mark current point (red circle with crosshair)
        current_pt = (int(all_com_history[-1][0]), int(all_com_history[-1][1]))
        cv2.circle(trail_frame, current_pt, 10, (0, 0, 255), -1)
        cv2.circle(trail_frame, current_pt, 12, (255, 255, 255), 2)
        cv2.line(trail_frame, (current_pt[0] - 15, current_pt[1]),
                 (current_pt[0] + 15, current_pt[1]), (0, 0, 255), 2)
        cv2.line(trail_frame, (current_pt[0], current_pt[1] - 15),
                 (current_pt[0], current_pt[1] + 15), (0, 0, 255), 2)
        cv2.putText(trail_frame, "CURRENT", (current_pt[0] + 15, current_pt[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return trail_frame


logger.info("Step 5-8 complete: COM calculation, smoothing, drawing, and trail functions ready.")


# ============================================================
# STEP 9: Process each frame
# ============================================================
frame_idx = 0
timestamp_ms = 0

# Storage
all_com_history = []           # All smoothed COM points (for full curve)
all_com_raw_history = []       # All raw COM points (for comparison)
annotations = []
frame_times = []               # Per-frame processing times

logger.info("Processing frames...")
logger.info("Step 9 started: entering frame-processing loop.")

while True:
    frame_start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect pose landmarks
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    if frame_idx == 0:
        logger.info("First frame processed by MediaPipe.")

    # Calculate COM if landmarks detected
    com_raw = None
    com_smoothed = None
    landmarks_data = None

    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        landmarks = detection_result.pose_landmarks[0]
        com_raw = calculate_com_raw(landmarks, width, height)
        com_smoothed = ema_smoother.update(com_raw[0], com_raw[1])

        # Store landmarks data for JSON
        landmarks_data = []
        for i, lm in enumerate(landmarks):
            landmarks_data.append({
                "index": i,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence
            })

    # Update histories
    if com_smoothed is not None:
        all_com_history.append(com_smoothed)
    if com_raw is not None:
        all_com_raw_history.append(com_raw)

    # Draw landmarks and COM (main video) - show both raw and smoothed
    annotated_frame = draw_landmarks_and_com(frame, detection_result, com_smoothed, com_raw)

    # Add frame number overlay
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, "Red=Smoothed COM, Orange=Raw COM", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw full COM trail curve + pose overlay (trail video)
    trail_frame = draw_full_trail_with_pose(frame, detection_result, all_com_history, frame_idx)
    cv2.putText(trail_frame, f"Frame: {frame_idx} / {total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(trail_frame, "Full COM Trajectory + Pose Overlay", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(trail_frame, f"Points: {len(all_com_history)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save annotated frame
    frame_filename = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(frame_filename, annotated_frame)

    # Write to output videos
    out.write(annotated_frame)
    out_trail.write(trail_frame)

    # Store annotation data for JSON
    frame_annotation = {
        "frame_index": frame_idx,
        "timestamp_ms": timestamp_ms,
        "com_raw": {"x": com_raw[0], "y": com_raw[1]} if com_raw else None,
        "com_smoothed": {"x": com_smoothed[0], "y": com_smoothed[1]} if com_smoothed else None,
        "landmarks": landmarks_data
    }
    annotations.append(frame_annotation)

    # Record frame processing time
    frame_end_time = time.perf_counter()
    frame_times.append(frame_end_time - frame_start_time)

    frame_idx += 1
    timestamp_ms = int(frame_idx * 1000 / fps)

    if frame_idx % 30 == 0:
        logger.info(
            "Processed %s/%s frames (timestamp=%sms, pose_detected=%s)...",
            frame_idx,
            total_frames,
            timestamp_ms,
            bool(detection_result.pose_landmarks),
        )


# ============================================================
# STEP 10: Compute and log timing statistics (.describe() style)
# ============================================================
frame_times_np = np.array(frame_times)
total_time = np.sum(frame_times_np)
avg_fps = frame_idx / total_time if total_time > 0 else 0

logger.info("=" * 60)
logger.info("FRAME PROCESSING TIME STATISTICS (seconds per frame)")
logger.info("=" * 60)
logger.info("Count:     %d", len(frame_times_np))
logger.info("Mean:      %.4f s (%.2f FPS)", np.mean(frame_times_np), 1/np.mean(frame_times_np) if np.mean(frame_times_np) > 0 else 0)
logger.info("Std:       %.4f s", np.std(frame_times_np))
logger.info("Min:       %.4f s", np.min(frame_times_np))
logger.info("25%%:       %.4f s", np.percentile(frame_times_np, 25))
logger.info("50%% (Med): %.4f s", np.percentile(frame_times_np, 50))
logger.info("75%%:       %.4f s", np.percentile(frame_times_np, 75))
logger.info("Max:       %.4f s", np.max(frame_times_np))
logger.info("Total:     %.2f s", total_time)
logger.info("Overall:   %.2f FPS effective", avg_fps)
logger.info("=" * 60)

# COM stability statistics
displacements = np.array([])
raw_displacements = np.array([])

if len(all_com_history) > 1:
    com_array = np.array(all_com_history)
    # Frame-to-frame displacement
    displacements = np.sqrt(np.sum(np.diff(com_array, axis=0)**2, axis=1))
    logger.info("COM STABILITY STATISTICS (smoothed, pixel displacement/frame)")
    logger.info("=" * 60)
    logger.info("Mean displacement:  %.2f px", np.mean(displacements))
    logger.info("Std displacement:   %.2f px", np.std(displacements))
    logger.info("Max displacement:   %.2f px", np.max(displacements))
    logger.info("Median displacement: %.2f px", np.median(displacements))
    logger.info("95th percentile:    %.2f px", np.percentile(displacements, 95))
    logger.info("Total path length:  %.2f px", np.sum(displacements))
    logger.info("=" * 60)

    # Raw vs smoothed comparison
    if len(all_com_raw_history) > 1:
        raw_array = np.array(all_com_raw_history)
        raw_displacements = np.sqrt(np.sum(np.diff(raw_array, axis=0)**2, axis=1))
        logger.info("RAW COM STATISTICS (for comparison)")
        logger.info("Mean displacement:  %.2f px", np.mean(raw_displacements))
        logger.info("Std displacement:   %.2f px", np.std(raw_displacements))
        logger.info("Max displacement:   %.2f px", np.max(raw_displacements))
        logger.info("Smoothing reduction: %.1f%%", (1 - np.mean(displacements)/np.mean(raw_displacements))*100 if np.mean(raw_displacements) > 0 else 0)
        logger.info("=" * 60)


# ============================================================
# STEP 11: Save JSON annotations
# ============================================================
json_data = {
    "video_info": {
        "input_video": INPUT_VIDEO,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": frame_idx
    },
    "com_settings": {
        "ema_alpha": EMA_ALPHA,
        "segment_weights": SEGMENT_WEIGHTS
    },
    "timing_stats": {
        "count": int(len(frame_times_np)),
        "mean_sec": float(np.mean(frame_times_np)),
        "std_sec": float(np.std(frame_times_np)),
        "min_sec": float(np.min(frame_times_np)),
        "max_sec": float(np.max(frame_times_np)),
        "median_sec": float(np.median(frame_times_np)),
        "total_sec": float(total_time),
        "effective_fps": float(avg_fps)
    },
    "com_stability_stats": {
        "mean_displacement_px": float(np.mean(displacements)) if len(displacements) > 0 else 0,
        "std_displacement_px": float(np.std(displacements)) if len(displacements) > 0 else 0,
        "max_displacement_px": float(np.max(displacements)) if len(displacements) > 0 else 0,
        "median_displacement_px": float(np.median(displacements)) if len(displacements) > 0 else 0,
        "total_path_length_px": float(np.sum(displacements)) if len(displacements) > 0 else 0
    },
    "frames": annotations
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(json_data, f, indent=2)
logger.info("Step 11 complete: JSON annotations saved to %s", OUTPUT_JSON)


# ============================================================
# STEP 12: Cleanup
# ============================================================
cap.release()
out.release()
out_trail.release()
detector.close()
logger.info("Step 12 complete: video capture, writers, and detector released.")

logger.info("Done! Processed %s frames.", frame_idx)
logger.info("Annotated frames saved to: %s/", OUTPUT_FRAMES_DIR)
logger.info("Output video (landmarks + raw/smoothed COM) saved to: %s", OUTPUT_VIDEO)
logger.info("Output video (FULL COM trajectory curve + pose overlay) saved to: %s", OUTPUT_VIDEO_TRAIL)
logger.info("JSON annotations saved to: %s", OUTPUT_JSON)