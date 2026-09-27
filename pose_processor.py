"""
Refactored Pose Processor for MediaPipe Bouldering Pose Analysis.
Converted from process_video.py into a class-based module with progress callbacks.
"""

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
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from threading import Thread
import queue


# ============================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# ============================================================

@dataclass
class ProcessingConfig:
    """Configuration for pose processing."""
    ema_alpha: float = 0.3
    trail_max_alpha: float = 0.8
    trail_min_alpha: float = 0.1
    segment_weights: Dict[str, float] = None
    com_landmark_indices: Dict[str, List[int]] = None
    save_frames: bool = True
    save_json: bool = True
    save_trail_video: bool = True
    segment_duration: int = 5  # Duration of preview segments in seconds

    def __post_init__(self):
        if self.segment_weights is None:
            self.segment_weights = {
                'head': 0.07,
                'torso': 0.43,
                'left_arm': 0.05,
                'right_arm': 0.05,
                'left_leg': 0.16,
                'right_leg': 0.16,
            }
        if self.com_landmark_indices is None:
            self.com_landmark_indices = {
                'head': [0],
                'torso': [11, 12, 23, 24],
                'left_arm': [13, 15],
                'right_arm': [14, 16],
                'left_leg': [25, 27],
                'right_leg': [26, 28],
            }


@dataclass
class FrameAnnotation:
    """Single frame annotation data."""
    frame_index: int
    timestamp_ms: int
    com_raw: Optional[Dict[str, float]]
    com_smoothed: Optional[Dict[str, float]]
    landmarks: Optional[List[Dict[str, float]]]


@dataclass
class ProcessingStats:
    """Processing timing and COM stability statistics."""
    frame_count: int
    total_time_sec: float
    effective_fps: float
    mean_frame_time_sec: float
    std_frame_time_sec: float
    min_frame_time_sec: float
    max_frame_time_sec: float
    median_frame_time_sec: float
    com_mean_displacement_px: float
    com_std_displacement_px: float
    com_max_displacement_px: float
    com_median_displacement_px: float
    com_total_path_length_px: float
    raw_com_mean_displacement_px: float = 0.0
    raw_com_std_displacement_px: float = 0.0
    raw_com_max_displacement_px: float = 0.0
    smoothing_reduction_pct: float = 0.0


@dataclass
class ProcessingResult:
    """Complete processing results."""
    success: bool
    output_dir: str
    output_video: str
    output_trail_video: str
    output_json: str
    frames_dir: str
    video_info: Dict[str, Any]
    stats: ProcessingStats
    error: Optional[str] = None
    segment_videos: List[str] = None
    segment_trail_videos: List[str] = None


# ============================================================
# PROGRESS CALLBACK TYPE
# ============================================================

ProgressCallback = Callable[[int, int, float, Optional[Dict], Optional[List[str]]], None]
# Args: current_frame, total_frames, current_fps, latest_com_data, completed_segments


# ============================================================
# POSE PROCESSOR CLASS
# ============================================================

class PoseProcessor:
    """
    MediaPipe Pose Landmarker processor for bouldering video analysis.
    Calculates Center of Mass (COM) with EMA smoothing and generates
    annotated output videos and JSON annotations.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    MODEL_FILENAME = "pose_landmarker.task"

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the pose processor.

        Args:
            config: ProcessingConfig object with all settings
            progress_callback: Function called with (current_frame, total_frames, fps, com_data)
            model_path: Path to MediaPipe model file (auto-downloads if not provided)
        """
        self.config = config or ProcessingConfig()
        self.progress_callback = progress_callback
        self.model_path = model_path or self.MODEL_FILENAME
        self.logger = self._setup_logger()
        self._detector = None
        self._ema_smoother = None
        self._cancelled = False

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this processor instance."""
        logger = logging.getLogger(f"PoseProcessor_{id(self)}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            ))
            logger.addHandler(handler)
        return logger

    def _ensure_model(self) -> str:
        """Download model if not exists, return path."""
        if not os.path.exists(self.model_path):
            self.logger.info("Downloading model to %s...", self.model_path)
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            self.logger.info("Download complete.")
        else:
            self.logger.info("Model already exists: %s", self.model_path)
        return self.model_path

    def _reencode_video_web_compatible(self, input_path: str, output_path: str) -> bool:
        """
        Re-encode video using ffmpeg for web-compatible H.264 playback.
        Uses baseline profile, yuv420p pixel format, and faststart for streaming.
        """
        try:
            # Check if ffmpeg is available
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("ffmpeg not available, skipping web re-encoding")
            return False

        try:
            # Re-encode with web-compatible settings:
            # -c:v libx264: Use H.264 encoder
            # -profile:v baseline: Baseline profile for maximum compatibility
            # -level 3.0: Level 3.0 for broad device support
            # -pix_fmt yuv420p: Required for browser playback
            # -movflags +faststart: Move moov atom to start for streaming
            # -preset fast: Reasonable encoding speed
            # -crf 23: Good quality/size balance
            cmd = [
                "ffmpeg", "-y",  # Overwrite output
                "-i", input_path,
                "-c:v", "libx264",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-preset", "fast",
                "-crf", "23",
                "-an",  # No audio
                output_path
            ]
            self.logger.info("Re-encoding video for web compatibility: %s", output_path)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            self.logger.info("Web re-encoding complete: %s", output_path)
            return True
        except subprocess.TimeoutExpired:
            self.logger.error("ffmpeg re-encoding timed out")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error("ffmpeg re-encoding failed: %s", e.stderr)
            return False
        except Exception as e:
            self.logger.error("Unexpected error during re-encoding: %s", e)
            return False

    def _split_video_into_segments(self, video_path: str, output_dir: str, segment_duration: int) -> list:
        """
        Split video into segments using ffmpeg.
        Returns list of segment file paths.
        """
        try:
            import ffmpeg
        except ImportError:
            self.logger.warning("ffmpeg-python not installed. Segment preview unavailable.")
            return []

        segments_dir = Path(output_dir)
        segments_dir.mkdir(exist_ok=True)

        # Get video duration
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            num_segments = int(duration / segment_duration) + 1
        except Exception as e:
            self.logger.error("Failed to probe video for segment splitting: %s", e)
            return []

        segments = []
        for i in range(num_segments):
            start_time = i * segment_duration
            segment_path = segments_dir / f"segment_{i + 1:03d}.mp4"

            try:
                (
                    ffmpeg
                    .input(video_path, ss=start_time, t=segment_duration)
                    .output(str(segment_path), c='copy', avoid_negative_ts='make_zero')
                    .overwrite_output()
                    .run(quiet=True)
                )
                if segment_path.exists() and segment_path.stat().st_size > 0:
                    segments.append(str(segment_path))
            except ffmpeg.Error:
                break

        return segments

    def _create_detector(self):
        """Create MediaPipe PoseLandmarker detector."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.VIDEO
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self.logger.info("PoseLandmarker created.")

    def _calculate_com_raw(self, landmarks, image_width: int, image_height: int) -> tuple:
        """Calculate raw Center of Mass from pose landmarks (no smoothing)."""
        com_x = 0.0
        com_y = 0.0
        total_weight = 0.0

        for segment_name, indices in self.config.com_landmark_indices.items():
            weight = self.config.segment_weights[segment_name]
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

    def _draw_landmarks_and_com(
        self,
        frame: np.ndarray,
        detection_result,
        com_point: Optional[tuple],
        com_raw: Optional[tuple] = None
    ) -> np.ndarray:
        """Draw pose landmarks and COM on the frame."""
        annotated_frame = frame.copy()

        pose_landmarks_list = detection_result.pose_landmarks
        if pose_landmarks_list:
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

        # Draw raw COM as small orange dot
        if com_raw is not None:
            rx, ry = int(com_raw[0]), int(com_raw[1])
            cv2.circle(annotated_frame, (rx, ry), 5, (0, 165, 255), -1)
            cv2.circle(annotated_frame, (rx, ry), 7, (255, 255, 255), 1)

        # Draw smoothed COM as red circle with crosshair
        if com_point is not None:
            com_x, com_y = int(com_point[0]), int(com_point[1])
            cv2.circle(annotated_frame, (com_x, com_y), 10, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (com_x, com_y), 12, (255, 255, 255), 2)
            cv2.line(annotated_frame, (com_x - 20, com_y), (com_x + 20, com_y), (0, 0, 255), 2)
            cv2.line(annotated_frame, (com_x, com_y - 20), (com_x, com_y + 20), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "COM (smoothed)", (com_x + 15, com_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return annotated_frame

    def _draw_full_trail_with_pose(
        self,
        frame: np.ndarray,
        detection_result,
        all_com_history: List[tuple],
        current_frame_idx: int
    ) -> np.ndarray:
        """Draw the full COM trajectory curve across all frames + current pose overlay."""
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
            for i in range(1, num_points):
                alpha_ratio = i / num_points

                pt1 = (int(all_com_history[i-1][0]), int(all_com_history[i-1][1]))
                pt2 = (int(all_com_history[i][0]), int(all_com_history[i][1]))

                # Color gradient: blue -> cyan -> green -> yellow -> red
                if alpha_ratio < 0.25:
                    t = alpha_ratio * 4
                    color = (int(255 * (1 - t)), int(255 * t), 255)
                elif alpha_ratio < 0.5:
                    t = (alpha_ratio - 0.25) * 4
                    color = (0, 255, int(255 * (1 - t)))
                elif alpha_ratio < 0.75:
                    t = (alpha_ratio - 0.5) * 4
                    color = (0, 255, int(255 * t))
                else:
                    t = (alpha_ratio - 0.75) * 4
                    color = (0, int(255 * (1 - t)), 255)

                alpha = self.config.trail_min_alpha + (self.config.trail_max_alpha - self.config.trail_min_alpha) * alpha_ratio
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

    def cancel(self):
        """Signal the processor to cancel."""
        self._cancelled = True

    def process_video(
        self,
        input_video_path: str,
        output_dir: str,
        progress_callback: Optional[ProgressCallback] = None
    ) -> ProcessingResult:
        """
        Process a video file and generate annotated outputs.

        Args:
            input_video_path: Path to input video file
            output_dir: Directory to save all outputs
            progress_callback: Optional callback(current_frame, total_frames, fps, com_data, completed_segments)

        Returns:
            ProcessingResult with all output paths and statistics
        """
        callback = progress_callback or self.progress_callback
        self._cancelled = False

        # Ensure model exists
        self._ensure_model()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frames_dir = output_path / "output_frames"
        if self.config.save_frames:
            frames_dir.mkdir(exist_ok=True)

        # Create segments directory for progressive preview
        segments_dir = output_path / "segments"
        segments_dir.mkdir(exist_ok=True)

        output_video_path = output_path / "output_video.mp4"
        output_trail_path = output_path / "output_video_com_trail.mp4"
        output_json_path = output_path / "pose_annotations.json"

        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return ProcessingResult(
                success=False,
                output_dir=str(output_path),
                output_video="",
                output_trail_video="",
                output_json="",
                frames_dir="",
                video_info={},
                stats=ProcessingStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                error=f"Could not open video: {input_video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info("Video info: %sx%s, %s FPS, %s frames", width, height, fps, total_frames)

        # Setup video writers - use H.264 (avc1) for browser compatibility
        # Try H.264 first, fall back to mp4v if not available
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        out_trail = cv2.VideoWriter(str(output_trail_path), fourcc, fps, (width, height))
        
        # Fallback to mp4v if H.264 not supported
        if not out.isOpened() or not out_trail.isOpened():
            self.logger.warning("H.264 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            out_trail = cv2.VideoWriter(str(output_trail_path), fourcc, fps, (width, height))

        if not out.isOpened() or not out_trail.isOpened():
            cap.release()
            return ProcessingResult(
                success=False,
                output_dir=str(output_path),
                output_video="",
                output_trail_video="",
                output_json="",
                frames_dir="",
                video_info={},
                stats=ProcessingStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                error="Could not open output video writers"
            )

        # Create detector
        self._create_detector()

        # Initialize EMA smoother
        class EMASmoother:
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

        self._ema_smoother = EMASmoother(alpha=self.config.ema_alpha)

        # Storage
        all_com_history = []
        all_com_raw_history = []
        annotations = []
        frame_times = []

        frame_idx = 0
        timestamp_ms = 0

        self.logger.info("Processing frames...")

        try:
            while True:
                if self._cancelled:
                    self.logger.info("Processing cancelled by user")
                    break

                frame_start_time = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Detect pose landmarks
                detection_result = self._detector.detect_for_video(mp_image, timestamp_ms)

                # Calculate COM if landmarks detected
                com_raw = None
                com_smoothed = None
                landmarks_data = None

                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    landmarks = detection_result.pose_landmarks[0]
                    com_raw = self._calculate_com_raw(landmarks, width, height)
                    com_smoothed = self._ema_smoother.update(com_raw[0], com_raw[1])

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

                # Draw landmarks and COM (main video)
                annotated_frame = self._draw_landmarks_and_com(frame, detection_result, com_smoothed, com_raw)

                # Add frame number overlay
                cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated_frame, "Red=Smoothed COM, Orange=Raw COM", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw full COM trail curve + pose overlay (trail video)
                trail_frame = self._draw_full_trail_with_pose(frame, detection_result, all_com_history, frame_idx)
                cv2.putText(trail_frame, f"Frame: {frame_idx} / {total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(trail_frame, "Full COM Trajectory + Pose Overlay", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(trail_frame, f"Points: {len(all_com_history)}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Save annotated frame
                if self.config.save_frames:
                    frame_filename = frames_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_filename), annotated_frame)

                # Write to output videos
                out.write(annotated_frame)
                out_trail.write(trail_frame)

                # Store annotation data for JSON
                frame_annotation = FrameAnnotation(
                    frame_index=frame_idx,
                    timestamp_ms=timestamp_ms,
                    com_raw={"x": com_raw[0], "y": com_raw[1]} if com_raw else None,
                    com_smoothed={"x": com_smoothed[0], "y": com_smoothed[1]} if com_smoothed else None,
                    landmarks=landmarks_data
                )
                annotations.append(frame_annotation)

                # Record frame processing time
                frame_end_time = time.perf_counter()
                frame_times.append(frame_end_time - frame_start_time)

                frame_idx += 1
                timestamp_ms = int(frame_idx * 1000 / fps)

                # Progress callback - update every frame for smooth progress bar
                if callback:
                    current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                    com_data = None
                    if com_smoothed:
                        com_data = {"x": com_smoothed[0], "y": com_smoothed[1]}
                    callback(frame_idx, total_frames, float(current_fps), com_data, [])

                if frame_idx % 30 == 0:
                    self.logger.info(
                        "Processed %s/%s frames (timestamp=%sms, pose_detected=%s)...",
                        frame_idx, total_frames, timestamp_ms, bool(detection_result.pose_landmarks)
                    )

        finally:
            # Cleanup
            cap.release()
            out.release()
            out_trail.release()
            if self._detector:
                self._detector.close()

        # Create segment videos by splitting the final output videos using ffmpeg
        # This ensures segments match the final video exactly (same as split_video_into_segments in app.py)
        completed_segments = []
        segment_trail_videos = []
        if self.config.segment_duration > 0:
            self.logger.info("Creating segment videos by splitting final output videos...")
            completed_segments = self._split_video_into_segments(str(output_video_path), str(segments_dir), self.config.segment_duration)
            segment_trail_videos = self._split_video_into_segments(str(output_trail_path), str(segments_dir), self.config.segment_duration)
            self.logger.info("Created %d main segments and %d trail segments", len(completed_segments), len(segment_trail_videos))

        # Compute statistics
        frame_times_np = np.array(frame_times)
        total_time = np.sum(frame_times_np)
        avg_fps = frame_idx / total_time if total_time > 0 else 0

        # COM stability statistics
        displacements = np.array([])
        raw_displacements = np.array([])

        if len(all_com_history) > 1:
            com_array = np.array(all_com_history)
            displacements = np.sqrt(np.sum(np.diff(com_array, axis=0)**2, axis=1))

            if len(all_com_raw_history) > 1:
                raw_array = np.array(all_com_raw_history)
                raw_displacements = np.sqrt(np.sum(np.diff(raw_array, axis=0)**2, axis=1))

        stats = ProcessingStats(
            frame_count=frame_idx,
            total_time_sec=float(total_time),
            effective_fps=float(avg_fps),
            mean_frame_time_sec=float(np.mean(frame_times_np)) if len(frame_times_np) > 0 else 0,
            std_frame_time_sec=float(np.std(frame_times_np)) if len(frame_times_np) > 0 else 0,
            min_frame_time_sec=float(np.min(frame_times_np)) if len(frame_times_np) > 0 else 0,
            max_frame_time_sec=float(np.max(frame_times_np)) if len(frame_times_np) > 0 else 0,
            median_frame_time_sec=float(np.median(frame_times_np)) if len(frame_times_np) > 0 else 0,
            com_mean_displacement_px=float(np.mean(displacements)) if len(displacements) > 0 else 0,
            com_std_displacement_px=float(np.std(displacements)) if len(displacements) > 0 else 0,
            com_max_displacement_px=float(np.max(displacements)) if len(displacements) > 0 else 0,
            com_median_displacement_px=float(np.median(displacements)) if len(displacements) > 0 else 0,
            com_total_path_length_px=float(np.sum(displacements)) if len(displacements) > 0 else 0,
            raw_com_mean_displacement_px=float(np.mean(raw_displacements)) if len(raw_displacements) > 0 else 0,
            raw_com_std_displacement_px=float(np.std(raw_displacements)) if len(raw_displacements) > 0 else 0,
            raw_com_max_displacement_px=float(np.max(raw_displacements)) if len(raw_displacements) > 0 else 0,
            smoothing_reduction_pct=float((1 - np.mean(displacements)/np.mean(raw_displacements))*100)
                if len(raw_displacements) > 0 and np.mean(raw_displacements) > 0 else 0
        )

        # Save JSON annotations
        if self.config.save_json:
            json_data = {
                "video_info": {
                    "input_video": input_video_path,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": total_frames,
                    "processed_frames": frame_idx
                },
                "com_settings": {
                    "ema_alpha": self.config.ema_alpha,
                    "segment_weights": self.config.segment_weights
                },
                "timing_stats": asdict(stats),
                "frames": [asdict(a) for a in annotations]
            }
            with open(output_json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            self.logger.info("JSON annotations saved to %s", output_json_path)
     
            self.logger.info("Done! Processed %s frames.", frame_idx)
            self.logger.info("Output video saved to: %s", output_video_path)
            self.logger.info("Trail video saved to: %s", output_trail_path)
     
            # Re-encode videos for web-compatible H.264 playback
            web_output_video = output_path / "output_video_web.mp4"
            web_output_trail = output_path / "output_video_com_trail_web.mp4"
             
            self.logger.info("Re-encoding videos for web compatibility...")
            video_reencoded = self._reencode_video_web_compatible(str(output_video_path), str(web_output_video))
            trail_reencoded = self._reencode_video_web_compatible(str(output_trail_path), str(web_output_trail))
             
            # Use web-compatible versions if re-encoding succeeded
            final_output_video = str(web_output_video) if video_reencoded else str(output_video_path)
            final_output_trail = str(web_output_trail) if trail_reencoded else str(output_trail_path)
             
            if video_reencoded:
                self.logger.info("Web-compatible video: %s", final_output_video)
            if trail_reencoded:
                self.logger.info("Web-compatible trail video: %s", final_output_trail)
     
            # Re-encode segment videos for web compatibility
            segment_videos = []
            segment_trail_videos_web = []
            
            # Re-encode main segment videos
            if completed_segments:
                self.logger.info("Re-encoding segment videos for web compatibility...")
                for i, seg_path in enumerate(completed_segments):
                    web_seg_path = Path(seg_path).with_name(Path(seg_path).stem + "_web.mp4")
                    seg_reencoded = self._reencode_video_web_compatible(seg_path, str(web_seg_path))
                    if seg_reencoded:
                        segment_videos.append(str(web_seg_path))
                        self.logger.info("Web-compatible segment %d: %s", i + 1, web_seg_path)
                    else:
                        segment_videos.append(seg_path)
            
            # Re-encode trail segment videos
            if segment_trail_videos:
                self.logger.info("Re-encoding segment trail videos for web compatibility...")
                for i, seg_trail_path in enumerate(segment_trail_videos):
                    web_seg_trail_path = Path(seg_trail_path).with_name(Path(seg_trail_path).stem + "_web.mp4")
                    seg_trail_reencoded = self._reencode_video_web_compatible(seg_trail_path, str(web_seg_trail_path))
                    if seg_trail_reencoded:
                        segment_trail_videos_web.append(str(web_seg_trail_path))
                        self.logger.info("Web-compatible segment trail %d: %s", i + 1, web_seg_trail_path)
                    else:
                        segment_trail_videos_web.append(seg_trail_path)
     
            return ProcessingResult(
                success=True,
                output_dir=str(output_path),
                output_video=final_output_video,
                output_trail_video=final_output_trail,
                output_json=str(output_json_path) if self.config.save_json else "",
                frames_dir=str(frames_dir) if self.config.save_frames else "",
                video_info={
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": total_frames,
                    "processed_frames": frame_idx
                },
                stats=stats,
                segment_videos=segment_videos,
                segment_trail_videos=segment_trail_videos_web
            )


# ============================================================
# ASYNC PROCESSING HELPER
# ============================================================

class AsyncPoseProcessor:
    """
    Wrapper to run PoseProcessor in a background thread with queue-based communication.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._thread: Optional[Thread] = None
        self._result_queue: queue.Queue = queue.Queue()
        self._progress_queue: queue.Queue = queue.Queue()
        self._processor: Optional[PoseProcessor] = None

    def start(self, input_video_path: str, output_dir: str):
        """Start processing in background thread."""
        def worker():
            def progress_cb(current, total, fps, com_data, completed_segments):
                self._progress_queue.put({
                    "current_frame": current,
                    "total_frames": total,
                    "current_fps": fps,
                    "com_data": com_data,
                    "completed_segments": completed_segments,
                    "progress_pct": (current / total * 100) if total > 0 else 0
                })

            self._processor = PoseProcessor(
                config=self.config,
                progress_callback=progress_cb
            )
            result = self._processor.process_video(input_video_path, output_dir)
            self._result_queue.put(result)

        self._thread = Thread(target=worker, daemon=True)
        self._thread.start()

    def get_progress(self) -> Optional[Dict]:
        """Get latest progress update (non-blocking)."""
        try:
            return self._progress_queue.get_nowait()
        except queue.Empty:
            return None

    def get_result(self, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Get processing result (blocking with optional timeout)."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """Check if processing thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    def cancel(self):
        """Cancel processing."""
        if self._processor:
            self._processor.cancel()

    def wait(self, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Wait for completion and return result."""
        if self._thread:
            self._thread.join(timeout=timeout)
        return self.get_result(timeout=0.1)