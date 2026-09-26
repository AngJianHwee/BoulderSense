!pip install mediapipe opencv-python 
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

import sys

# 1. Clear any pre-existing handlers that Colab sets up automatically
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 2. Re-configure your logging settings cleanly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout # Ensures logs route directly to your cell output
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

# COM Trail settings
TRAIL_LENGTH = 60  # Number of frames to show in trail
TRAIL_MAX_ALPHA = 0.8
TRAIL_MIN_ALPHA = 0.1

# Create output directory
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
logger.info("Step 0 complete: output directory ready: %s", OUTPUT_FRAMES_DIR)

# download if not exists
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

# Second video writer for COM trail
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
# MediaPipe Pose has 33 landmarks. We'll use key body landmarks.
# Using standard body segment weights for COM approximation
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

# Simplified COM using major body landmarks with approximate weights
# Head (nose), Torso (shoulders, hips), Arms, Legs
COM_LANDMARK_INDICES = {
    'head': [0],                    # nose as head proxy
    'torso': [11, 12, 23, 24],     # shoulders and hips
    'left_arm': [13, 15],           # left elbow, wrist
    'right_arm': [14, 16],          # right elbow, wrist
    'left_leg': [25, 27],           # left knee, ankle
    'right_leg': [26, 28],          # right knee, ankle
}

# Approximate body segment weights (percentage of total body mass)
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
# STEP 5: Function to calculate Center of Mass
# ============================================================
def calculate_com(landmarks, image_width, image_height):
    """
    Calculate Center of Mass from pose landmarks.
    Returns (x, y) in pixel coordinates.
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
                # Only use landmarks with good visibility
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
    
    return int(com_x), int(com_y)

# ============================================================
# STEP 6: Function to draw landmarks and COM on frame
# ============================================================
def draw_landmarks_and_com(frame, detection_result, com_point):
    """
    Draw pose landmarks and COM on the frame.
    """
    annotated_frame = frame.copy()
    
    # Draw pose landmarks
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    
    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_frame,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)
    
    # Draw Center of Mass as a red circle with crosshair
    if com_point is not None:
        com_x, com_y = com_point
        # Red circle
        cv2.circle(annotated_frame, (com_x, com_y), 10, (0, 0, 255), -1)
        # White border
        cv2.circle(annotated_frame, (com_x, com_y), 12, (255, 255, 255), 2)
        # Crosshair
        cv2.line(annotated_frame, (com_x - 20, com_y), (com_x + 20, com_y), (0, 0, 255), 2)
        cv2.line(annotated_frame, (com_x, com_y - 20), (com_x, com_y + 20), (0, 0, 255), 2)
        # Label
        cv2.putText(annotated_frame, "COM", (com_x + 15, com_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return annotated_frame

# ============================================================
# STEP 6b: Function to draw COM trail (gradient)
# ============================================================
def draw_com_trail(frame, com_history, trail_length=TRAIL_LENGTH):
    """
    Draw COM trail with gradient effect (older = more transparent).
    """
    trail_frame = frame.copy()
    num_points = len(com_history)
    
    if num_points < 2:
        return trail_frame
    
    # Draw trail segments with gradient alpha
    for i in range(1, num_points):
        # Calculate alpha based on position in trail (newer = more opaque)
        alpha_ratio = i / num_points
        alpha = TRAIL_MIN_ALPHA + (TRAIL_MAX_ALPHA - TRAIL_MIN_ALPHA) * alpha_ratio
        
        pt1 = com_history[i - 1]
        pt2 = com_history[i]
        
        # Color gradient: blue (old) to red (new)
        blue = int(255 * (1 - alpha_ratio))
        red = int(255 * alpha_ratio)
        color = (blue, 0, red)
        
        # Line thickness decreases for older segments
        thickness = max(1, int(4 * alpha_ratio))
        
        # Create overlay for alpha blending
        overlay = trail_frame.copy()
        cv2.line(overlay, pt1, pt2, color, thickness)
        cv2.addWeighted(overlay, alpha, trail_frame, 1 - alpha, 0, trail_frame)
    
    # Draw current COM position as a bright marker
    if com_history:
        current_com = com_history[-1]
        cv2.circle(trail_frame, current_com, 8, (0, 255, 255), -1)  # Yellow center
        cv2.circle(trail_frame, current_com, 10, (255, 255, 255), 2)  # White border
    
    return trail_frame

logger.info("Step 5-6b complete: COM calculation, drawing, and trail functions ready.")

# ============================================================
# STEP 7: Process each frame
# ============================================================
frame_idx = 0
timestamp_ms = 0

# Storage for COM history (for trail) and JSON annotations
com_history = []
annotations = []

logger.info("Processing frames...")
logger.info("Step 7 started: entering frame-processing loop.")

while True:
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
    com_point = None
    landmarks_data = None
    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        landmarks = detection_result.pose_landmarks[0]
        com_point = calculate_com(landmarks, width, height)
        
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
    
    # Update COM history for trail
    if com_point is not None:
        com_history.append(com_point)
        if len(com_history) > TRAIL_LENGTH:
            com_history.pop(0)
    
    # Draw landmarks and COM (main video)
    annotated_frame = draw_landmarks_and_com(frame, detection_result, com_point)
    
    # Add frame number overlay
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw COM trail (trail video)
    trail_frame = draw_com_trail(frame, com_history)
    cv2.putText(trail_frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(trail_frame, "COM Trail", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
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
        "com": {"x": com_point[0], "y": com_point[1]} if com_point else None,
        "landmarks": landmarks_data
    }
    annotations.append(frame_annotation)
    
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
# STEP 8: Save JSON annotations
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
        "trail_length": TRAIL_LENGTH,
        "segment_weights": SEGMENT_WEIGHTS
    },
    "frames": annotations
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(json_data, f, indent=2)
logger.info("Step 8 complete: JSON annotations saved to %s", OUTPUT_JSON)

# ============================================================
# STEP 9: Cleanup
# ============================================================
cap.release()
out.release()
out_trail.release()
detector.close()
logger.info("Step 9 complete: video capture, writers, and detector released.")

logger.info("Done! Processed %s frames.", frame_idx)
logger.info("Annotated frames saved to: %s/", OUTPUT_FRAMES_DIR)
logger.info("Output video (landmarks + COM) saved to: %s", OUTPUT_VIDEO)
logger.info("Output video (COM trail) saved to: %s", OUTPUT_VIDEO_TRAIL)
logger.info("JSON annotations saved to: %s", OUTPUT_JSON)