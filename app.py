"""
Streamlit GUI for MediaPipe Bouldering Pose Analysis.
Upload a video, process it with pose detection, and view results progressively.
"""

import streamlit as st
import tempfile
import os
import json
import time
import subprocess
import threading
import requests
import cv2
from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our processor
from pose_processor import (
    AsyncPoseProcessor,
    ProcessingConfig,
    ProcessingResult,
    ProcessingStats
)


# ============================================================
# NGROK TUNNEL INTEGRATION
# ============================================================

def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def start_ngrok_tunnel(port: int = 8501) -> Optional[str]:
    """
    Start ngrok tunnel for the given port.
    Returns public URL if successful, None otherwise.
    """
    load_env()
    
    token = os.environ.get("NGROK_AUTH_TOKEN")
    if not token:
        return None
    
    # Get custom domain from env
    custom_domain = os.environ.get("NGROK_DOMAIN")
    
    # Check if ngrok is installed
    try:
        subprocess.run(["ngrok", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    
    # Configure ngrok with auth token
    try:
        subprocess.run(["ngrok", "config", "add-authtoken", token],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return None
    
    # Start tunnel in background
    try:
        # Build ngrok command
        ngrok_cmd = ["ngrok", "http", str(port), "--log=stdout"]
        if custom_domain:
            ngrok_cmd.extend(["--domain", custom_domain])
        
        process = subprocess.Popen(
            ngrok_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for tunnel to be ready
        time.sleep(3)
        
        # Get public URL from ngrok API
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
            tunnels = response.json().get("tunnels", [])
            for tunnel in tunnels:
                if tunnel["proto"] == "https":
                    return tunnel["public_url"]
        except requests.RequestException:
            pass
        
        process.terminate()
        return None
        
    except Exception:
        return None


def get_tunnel_url() -> Optional[str]:
    """Get the current active tunnel URL."""
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=3)
        tunnels = response.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
    except Exception:
        pass
    return None


def init_ngrok_tunnel():
    """Initialize ngrok tunnel if auth token is configured."""
    if "ngrok_url" not in st.session_state:
        st.session_state.ngrok_url = None
        st.session_state.ngrok_started = False
    
    # Only start if token exists and not already started
    if not st.session_state.ngrok_started and os.environ.get("NGROK_AUTH_TOKEN"):
        st.session_state.ngrok_started = True
        
        # Start tunnel in background thread
        def start_tunnel():
            url = start_ngrok_tunnel(8501)
            if url:
                st.session_state.ngrok_url = url
                print(f"\n🌐 Ngrok tunnel active: {url}\n", flush=True)
        
        thread = threading.Thread(target=start_tunnel, daemon=True)
        thread.start()
    
    # Also check for existing tunnel
    if not st.session_state.ngrok_url:
        existing_url = get_tunnel_url()
        if existing_url:
            st.session_state.ngrok_url = existing_url
            print(f"\n🌐 Ngrok tunnel active: {existing_url}\n", flush=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bouldering Pose Analysis",
    page_icon="🧗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stVideo > div {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .progress-container {
        background: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .segment-list {
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "processing": False,
        "processor": None,
        "result": None,
        "progress": {"current_frame": 0, "total_frames": 0, "current_fps": 0, "com_data": None, "progress_pct": 0},
        "uploaded_file": None,
        "temp_dir": None,
        "output_dir": None,
        "segments": [],
        "active_tab": "preview",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return path."""
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="bouldering_")
    temp_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def create_output_dir() -> str:
    """Create output directory for processing results."""
    if st.session_state.output_dir is None:
        st.session_state.output_dir = tempfile.mkdtemp(prefix="bouldering_output_")
    return st.session_state.output_dir


def split_video_into_segments(video_path: str, output_dir: str, segment_duration: int = 5) -> list:
    """
    Split video into segments using ffmpeg.
    Returns list of segment file paths.
    """
    try:
        import ffmpeg
    except ImportError:
        st.warning("ffmpeg-python not installed. Segment preview unavailable.")
        return []

    segments_dir = Path(output_dir) / "segments"
    segments_dir.mkdir(exist_ok=True)

    # Get video duration
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    num_segments = int(duration / segment_duration) + 1

    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        segment_path = segments_dir / f"segment_{i:03d}.mp4"

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


def get_latest_segment(segments: list) -> Optional[str]:
    """Get the latest segment file path."""
    if segments:
        return segments[-1]
    return None


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def create_com_displacement_chart(stats: ProcessingStats, annotations: list) -> go.Figure:
    """Create Plotly chart for COM displacement over time."""
    if not annotations:
        return go.Figure()

    # Extract COM data from annotations
    frames = []
    com_x = []
    com_y = []
    timestamps = []

    for a in annotations:
        if a.get('com_smoothed'):
            frames.append(a['frame_index'])
            com_x.append(a['com_smoothed']['x'])
            com_y.append(a['com_smoothed']['y'])
            timestamps.append(a['timestamp_ms'] / 1000)

    if not frames:
        return go.Figure()

    # Calculate frame-to-frame displacement
    displacements = [0]
    for i in range(1, len(com_x)):
        dx = com_x[i] - com_x[i-1]
        dy = com_y[i] - com_y[i-1]
        displacements.append((dx**2 + dy**2)**0.5)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("COM Trajectory (X vs Y)", "Frame-to-Frame Displacement"),
        vertical_spacing=0.15,
        shared_xaxes=True
    )

    # Trajectory plot
    fig.add_trace(
        go.Scatter(
            x=com_x, y=com_y,
            mode='lines+markers',
            marker=dict(size=4, color=timestamps, colorscale='Viridis', showscale=True, colorbar=dict(title="Time (s)")),
            line=dict(width=1, color='rgba(31, 119, 180, 0.5)'),
            name='COM Path',
            hovertemplate='Frame: %{customdata[0]}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Time: %{customdata[1]:.1f}s<extra></extra>',
            customdata=list(zip(frames, timestamps))
        ),
        row=1, col=1
    )

    # Mark start and end
    fig.add_trace(
        go.Scatter(x=[com_x[0]], y=[com_y[0]], mode='markers', marker=dict(size=12, color='green', symbol='circle'),
                   name='Start', hovertemplate='Start<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[com_x[-1]], y=[com_y[-1]], mode='markers', marker=dict(size=12, color='red', symbol='circle'),
                   name='End', hovertemplate='End<extra></extra>'),
        row=1, col=1
    )

    # Displacement plot
    fig.add_trace(
        go.Scatter(
            x=frames, y=displacements,
            mode='lines',
            line=dict(width=1, color='red'),
            name='Displacement (px/frame)',
            hovertemplate='Frame: %{x}<br>Displacement: %{y:.2f} px<extra></extra>'
        ),
        row=2, col=1
    )

    # Add mean line
    mean_disp = stats.com_mean_displacement_px
    fig.add_hline(y=mean_disp, line_dash="dash", line_color="orange",
                  annotation_text=f"Mean: {mean_disp:.2f} px", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Center of Mass Analysis",
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)",
        xaxis2_title="Frame Index",
        yaxis2_title="Displacement (px)"
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)  # Image coordinates

    return fig


def create_timing_chart(frame_times: list) -> go.Figure:
    """Create Plotly chart for frame processing times."""
    if not frame_times:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Frame Processing Time", "Processing Time Distribution"),
        vertical_spacing=0.15
    )

    # Time series
    fig.add_trace(
        go.Scatter(
            x=list(range(len(frame_times))),
            y=frame_times,
            mode='lines',
            line=dict(width=0.5, color='blue'),
            name='Frame Time (s)',
            hovertemplate='Frame: %{x}<br>Time: %{y:.4f}s<extra></extra>'
        ),
        row=1, col=1
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=frame_times,
            nbinsx=50,
            marker_color='lightblue',
            name='Distribution',
            hovertemplate='Time: %{x:.4f}s<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Frame Index", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Frame Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


# ============================================================
# FRAGMENT FOR LIVE PREVIEW (POLLS EVERY 2 SECONDS)
# ============================================================

@st.fragment(run_every=2)
def live_preview_fragment():
    """Fragment that polls for latest progress and displays input video segments during processing."""
    if not st.session_state.processing:
        return

    # Poll processor for latest progress
    processor = st.session_state.get("processor")
    if processor:
        # Get all pending progress updates
        while True:
            progress = processor.get_progress()
            if progress is None:
                break
            st.session_state.progress = progress
        
        # Check for result
        result = processor.get_result(timeout=0.1)
        if result is not None:
            st.session_state.processing = False
            st.session_state.result = result
            st.session_state.processor = None

            # Store output segments separately (don't replace input segments yet)
            if result.success and result.output_video:
                with st.spinner("Generating output segment previews..."):
                    segment_duration = st.session_state.get("segment_duration", 5)
                    output_segments = split_video_into_segments(
                        result.output_video,
                        result.output_dir,
                        segment_duration=segment_duration
                    )
                    st.session_state.output_segments = output_segments

            st.rerun()

    progress = st.session_state.progress
    # Use input segments during processing, output segments after completion
    segments = st.session_state.get("input_segments", st.session_state.get("segments", []))

    # Progress bar with text
    progress_pct = progress.get("progress_pct", 0)
    current_frame = progress.get("current_frame", 0)
    total_frames = progress.get("total_frames", 0)
    current_fps = progress.get("current_fps", 0)
    
    # Prominent progress bar
    st.progress(progress_pct / 100, text=f"🔄 Processing: {progress_pct:.1f}% ({current_frame}/{total_frames} frames) @ {current_fps:.1f} FPS")
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames", f"{current_frame} / {total_frames}")
    with col2:
        st.metric("Processing FPS", f"{current_fps:.1f}")
    with col3:
        st.metric("Progress", f"{progress_pct:.1f}%")
    with col4:
        elapsed = current_frame / current_fps if current_fps > 0 else 0
        remaining = (total_frames - current_frame) / current_fps if current_fps > 0 else 0
        st.metric("Est. Remaining", f"{remaining:.0f}s")

    # Latest segment preview (from input video during processing)
    latest_seg = get_latest_segment(segments)
    if latest_seg and os.path.exists(latest_seg):
        segment_duration = st.session_state.get("segment_duration", 5)
        st.subheader(f"📹 Latest {segment_duration}-Second Segment (Input Preview)")
        st.video(latest_seg)
        st.caption(f"Segment: {os.path.basename(latest_seg)}")

    # COM data display
    com_data = progress.get("com_data")
    if com_data:
        st.subheader("📍 Current COM Position")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("X (px)", f"{com_data['x']:.1f}")
        with c2:
            st.metric("Y (px)", f"{com_data['y']:.1f}")

    # All segments list
    if segments:
        segment_duration = st.session_state.get("segment_duration", 5)
        with st.expander(f"📁 All {segment_duration}-Second Segments ({len(segments)})", expanded=False):
            for i, seg in enumerate(segments):
                if os.path.exists(seg):
                    st.video(seg)
                    st.caption(f"Segment {i}: {os.path.basename(seg)}")


# ============================================================
# MAIN APP LAYOUT
# ============================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🧗 Bouldering Pose Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a climbing video to analyze Center of Mass trajectory and pose stability</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📤 Upload & Settings")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload a bouldering/climbing video for pose analysis"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.video(uploaded_file)

        st.divider()

        # Processing settings
        with st.expander("⚙️ Processing Settings", expanded=True):
            ema_alpha = st.slider(
                "EMA Smoothing Alpha",
                min_value=0.1, max_value=0.5, value=0.3, step=0.05,
                help="Lower = more smoothing, higher = more responsive"
            )

            save_frames = st.checkbox("Save annotated frames", value=True)
            save_json = st.checkbox("Save JSON annotations", value=True)
            save_trail = st.checkbox("Save trail video (full COM trajectory)", value=True)

        st.divider()

        # Segment settings
        with st.expander("🎬 Segment Preview Settings", expanded=False):
            segment_duration = st.number_input(
                "Segment Duration (seconds)",
                min_value=1, max_value=30, value=5, step=1
            )
            st.caption("Videos will be split into segments of this duration for progressive preview")

        st.divider()

        # Start button
        if st.session_state.uploaded_file and not st.session_state.processing:
            if st.button("🚀 Start Processing", type="primary", width="stretch"):
                start_processing(ema_alpha, save_frames, save_json, save_trail, segment_duration)

        # Cancel button
        if st.session_state.processing:
            if st.button("⏹️ Cancel Processing", type="secondary", width="stretch"):
                cancel_processing()

        # Status display - progress is handled by live_preview_fragment
        if st.session_state.processing:
            st.info("⏳ Processing in progress... See Live Preview tab for real-time updates.")
        elif st.session_state.result and st.session_state.result.success:
            st.success("✅ Processing complete!")
        elif st.session_state.result and not st.session_state.result.success:
            st.error(f"❌ Error: {st.session_state.result.error}")

        # Ngrok tunnel status
        st.divider()
        st.header("🌐 Public Tunnel")
        if st.session_state.get("ngrok_url"):
            st.success("✅ Tunnel Active")
            st.code(st.session_state.ngrok_url)
            st.caption("Share this URL to access the app remotely")
            st.link_button("🔗 Open in Browser", st.session_state.ngrok_url)
        elif os.environ.get("NGROK_AUTH_TOKEN"):
            st.info("🔄 Starting tunnel...")
            st.caption("Ngrok auth token found, establishing tunnel...")
        else:
            st.warning("⚠️ No tunnel configured")
            st.caption("Set NGROK_AUTH_TOKEN in .env to enable public access")

    # Main content area - Tabs
    tab_preview, tab_results, tab_analytics, tab_config = st.tabs([
        "📹 Live Preview",
        "📊 Results",
        "📈 Analytics",
        "⚙️ Configuration"
    ])

    with tab_preview:
        render_preview_tab()

    with tab_results:
        render_results_tab()

    with tab_analytics:
        render_analytics_tab()

    with tab_config:
        render_config_tab()


def start_processing(ema_alpha, save_frames, save_json, save_trail, segment_duration):
    """Start the background processing."""
    # Save uploaded file
    input_path = save_uploaded_file(st.session_state.uploaded_file)
    output_dir = create_output_dir()

    # Create config
    config = ProcessingConfig(
        ema_alpha=ema_alpha,
        save_frames=save_frames,
        save_json=save_json,
        save_trail_video=save_trail
    )

    # Create async processor
    processor = AsyncPoseProcessor(config=config)
    processor.start(input_path, output_dir)

    # Pre-split INPUT video into segments for preview during processing
    # This gives users something to watch while processing happens
    with st.spinner("Preparing segment previews..."):
        input_segments = split_video_into_segments(
            input_path,
            output_dir,
            segment_duration=segment_duration
        )
        st.session_state.segments = input_segments
        st.session_state.input_segments = input_segments  # Keep reference to input segments

    # Update session state
    st.session_state.processing = True
    st.session_state.processor = processor
    st.session_state.result = None
    # Initialize progress with total_frames from video to avoid 0/0 display
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    st.session_state.progress = {"current_frame": 0, "total_frames": total_frames, "current_fps": 0, "com_data": None, "progress_pct": 0}
    
    # Store settings for Configuration tab display
    st.session_state.ema_alpha = ema_alpha
    st.session_state.save_frames = save_frames
    st.session_state.save_json = save_json
    st.session_state.save_trail = save_trail
    st.session_state.segment_duration = segment_duration

    # Trigger rerun to start polling
    st.rerun()


def cancel_processing():
    """Cancel the background processing."""
    if st.session_state.processor:
        st.session_state.processor.cancel()
    st.session_state.processing = False
    st.session_state.processor = None
    st.rerun()


def render_preview_tab():
    """Render the Live Preview tab."""
    st.header("📹 Live Preview")

    if st.session_state.processing:
        # Show prominent progress bar at top of preview tab
        progress = st.session_state.progress
        progress_pct = progress.get("progress_pct", 0)
        current_frame = progress.get("current_frame", 0)
        total_frames = progress.get("total_frames", 0)
        current_fps = progress.get("current_fps", 0)
        
        # Large progress bar with detailed text
        st.progress(progress_pct / 100, text=f"🔄 Processing: {progress_pct:.1f}% ({current_frame}/{total_frames} frames) @ {current_fps:.1f} FPS")
        
        # Additional progress metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Frames Processed", f"{current_frame} / {total_frames}")
        with col2:
            st.metric("Processing FPS", f"{current_fps:.1f}")
        with col3:
            st.metric("Progress", f"{progress_pct:.1f}%")
        with col4:
            elapsed = current_frame / current_fps if current_fps > 0 else 0
            remaining = (total_frames - current_frame) / current_fps if current_fps > 0 else 0
            st.metric("Est. Remaining", f"{remaining:.0f}s")
        
        st.info("⏳ Processing in progress... Live preview updates in real-time.")
        live_preview_fragment()

    elif st.session_state.result and st.session_state.result.success:
        st.success("✅ Processing complete! View results in the **Results** tab.")
        result = st.session_state.result

        # Show final videos
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Main Output (Landmarks + COM)")
            if os.path.exists(result.output_video):
                st.video(result.output_video)
            else:
                st.warning("Video file not found")

        with col2:
            st.subheader("Trail Video (Full COM Trajectory)")
            if os.path.exists(result.output_trail_video):
                st.video(result.output_trail_video)
            else:
                st.warning("Trail video not found")

        # Segments - show output segments if available, otherwise input segments
        output_segments = st.session_state.get("output_segments", [])
        input_segments = st.session_state.get("input_segments", [])
        segments = output_segments if output_segments else input_segments
        
        if segments:
            segment_duration = st.session_state.get("segment_duration", 5)
            label = "Output" if output_segments else "Input"
            st.subheader(f"📁 All {segment_duration}-Second Segments ({label})")
            for i, seg in enumerate(segments):
                if os.path.exists(seg):
                    with st.expander(f"Segment {i}: {os.path.basename(seg)}"):
                        st.video(seg)

    else:
        st.info("👈 Upload a video and click **Start Processing** to begin")


def render_results_tab():
    """Render the Results tab."""
    st.header("📊 Results")

    if not st.session_state.result or not st.session_state.result.success:
        st.info("No results yet. Process a video first.")
        return

    result: ProcessingResult = st.session_state.result
    stats: ProcessingStats = result.stats

    # Video players
    st.subheader("🎬 Output Videos")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Main Output** (Pose landmarks + Raw/Smoothed COM)")
        if os.path.exists(result.output_video):
            st.video(result.output_video)
            st.download_button(
                "⬇️ Download Main Video",
                data=open(result.output_video, "rb").read(),
                file_name="output_video.mp4",
                mime="video/mp4"
            )
        else:
            st.warning("Video not found")

    with col2:
        st.markdown("**Trail Video** (Full COM trajectory + pose overlay)")
        if os.path.exists(result.output_trail_video):
            st.video(result.output_trail_video)
            st.download_button(
                "⬇️ Download Trail Video",
                data=open(result.output_trail_video, "rb").read(),
                file_name="output_video_com_trail.mp4",
                mime="video/mp4"
            )
        else:
            st.warning("Trail video not found")

    # Output segments
    output_segments = st.session_state.get("output_segments", [])
    if output_segments:
        segment_duration = st.session_state.get("segment_duration", 5)
        st.subheader(f"📁 Output Segments ({segment_duration}s each)")
        for i, seg in enumerate(output_segments):
            if os.path.exists(seg):
                with st.expander(f"Segment {i}: {os.path.basename(seg)}"):
                    st.video(seg)

    st.divider()

    # Statistics
    st.subheader("📈 Processing Statistics")

    # Timing stats
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.metric("Frames Processed", stats.frame_count)
    # with col2:
    #     st.metric("Total Time", f"{stats.total_time_sec:.1f}s")
    # with col3:
    #     st.metric("Effective FPS", f"{stats.effective_fps:.1f}")
    # with col4:
    #     st.metric("Avg Frame Time", f"{stats.mean_frame_time_sec*1000:.1f}ms")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Frame Time", f"{stats.min_frame_time_sec*1000:.1f}ms")
    with col2:
        st.metric("Max Frame Time", f"{stats.max_frame_time_sec*1000:.1f}ms")
    with col3:
        st.metric("Median Frame Time", f"{stats.median_frame_time_sec*1000:.1f}ms")
    with col4:
        st.metric("Std Dev", f"{stats.std_frame_time_sec*1000:.1f}ms")

    st.divider()

    # COM Stability stats
    st.subheader("🎯 COM Stability Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Displacement", f"{stats.com_mean_displacement_px:.2f} px/frame")
    with col2:
        st.metric("Max Displacement", f"{stats.com_max_displacement_px:.2f} px/frame")
    with col3:
        st.metric("Median Displacement", f"{stats.com_median_displacement_px:.2f} px/frame")
    with col4:
        st.metric("Total Path Length", f"{stats.com_total_path_length_px:.1f} px")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Std Deviation", f"{stats.com_std_displacement_px:.2f} px")
    with col2:
        st.metric("Smoothing Reduction", f"{stats.smoothing_reduction_pct:.1f}%")
    with col3:
        st.metric("Raw Mean Displacement", f"{stats.raw_com_mean_displacement_px:.2f} px")

    st.divider()

    # JSON download
    st.subheader("📄 JSON Annotations")
    if result.output_json and os.path.exists(result.output_json):
        with open(result.output_json, "r") as f:
            json_data = json.load(f)

        st.download_button(
            "⬇️ Download Full JSON",
            data=json.dumps(json_data, indent=2),
            file_name="pose_annotations.json",
            mime="application/json"
        )

        # Show summary
        with st.expander("View JSON Summary", expanded=False):
            st.json({
                "video_info": json_data.get("video_info", {}),
                "com_settings": json_data.get("com_settings", {}),
                "timing_stats": json_data.get("timing_stats", {}),
                "frame_count": len(json_data.get("frames", []))
            })

    # Frames directory info
    if result.frames_dir and os.path.exists(result.frames_dir):
        frame_count = len(list(Path(result.frames_dir).glob("*.jpg")))
        st.info(f"📁 {frame_count} annotated frames saved to: `{result.frames_dir}`")


def render_analytics_tab():
    """Render the Analytics tab with charts."""
    st.header("📈 Analytics")

    if not st.session_state.result or not st.session_state.result.success:
        st.info("No analytics yet. Process a video first.")
        return

    result: ProcessingResult = st.session_state.result

    # Load annotations for charts
    annotations = []
    frame_times = []

    if result.output_json and os.path.exists(result.output_json):
        with open(result.output_json, "r") as f:
            json_data = json.load(f)
        annotations = json_data.get("frames", [])
        # Frame times not stored in JSON, would need to be added

    # COM Trajectory Chart
    st.subheader("🎯 Center of Mass Trajectory")
    fig_com = create_com_displacement_chart(result.stats, annotations)
    st.plotly_chart(fig_com, width="stretch")

    st.divider()

    # Video info
    st.subheader("📹 Video Information")
    vinfo = result.video_info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Resolution", f"{vinfo.get('width', 0)}x{vinfo.get('height', 0)}")
    with col2:
        st.metric("FPS", f"{vinfo.get('fps', 0):.1f}")
    with col3:
        st.metric("Total Frames", vinfo.get('total_frames', 0))
    with col4:
        st.metric("Processed Frames", vinfo.get('processed_frames', 0))

    st.divider()

    # Per-frame data table
    st.subheader("📋 Per-Frame Data")
    if annotations:
        # Create DataFrame for display
        df_data = []
        for a in annotations:
            row = {
                "Frame": a["frame_index"],
                "Time (s)": a["timestamp_ms"] / 1000,
                "COM X (smoothed)": a["com_smoothed"]["x"] if a["com_smoothed"] else None,
                "COM Y (smoothed)": a["com_smoothed"]["y"] if a["com_smoothed"] else None,
                "COM X (raw)": a["com_raw"]["x"] if a["com_raw"] else None,
                "COM Y (raw)": a["com_raw"]["y"] if a["com_raw"] else None,
                "Landmarks": len(a["landmarks"]) if a["landmarks"] else 0
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        st.dataframe(df, width="stretch", height=400)

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download as CSV",
            data=csv,
            file_name="frame_annotations.csv",
            mime="text/csv"
        )
    else:
        st.info("No per-frame data available")


def render_config_tab():
    """Render the Configuration tab showing current settings."""
    st.header("⚙️ Current Configuration")
    
    # Processing settings
    st.subheader("🎯 Processing Settings")
    
    # Get current settings from session state or defaults
    ema_alpha = st.session_state.get("ema_alpha", 0.3)
    save_frames = st.session_state.get("save_frames", True)
    save_json = st.session_state.get("save_json", True)
    save_trail = st.session_state.get("save_trail", True)
    segment_duration = st.session_state.get("segment_duration", 5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("EMA Smoothing Alpha", f"{ema_alpha}")
        st.caption("Lower = more smoothing, higher = more responsive")
        
        st.metric("Save Annotated Frames", "✅ Yes" if save_frames else "❌ No")
        st.metric("Save JSON Annotations", "✅ Yes" if save_json else "❌ No")
        
    with col2:
        st.metric("Save Trail Video", "✅ Yes" if save_trail else "❌ No")
        st.metric("Segment Duration", f"{segment_duration}s")
        st.caption("Duration for progressive preview segments")
    
    st.divider()
    
    # COM Calculation Settings
    st.subheader("📍 Center of Mass Calculation")
    
    # Segment weights
    segment_weights = {
        'head': 0.07,
        'torso': 0.43,
        'left_arm': 0.05,
        'right_arm': 0.05,
        'left_leg': 0.16,
        'right_leg': 0.16,
    }
    
    st.markdown("**Segment Weights (for COM calculation):**")
    weight_df = pd.DataFrame([
        {"Segment": k.replace("_", " ").title(), "Weight": f"{v*100:.0f}%"}
        for k, v in segment_weights.items()
    ])
    st.dataframe(weight_df, width="stretch", hide_index=True)
    
    st.caption("Based on biomechanical body segment parameters (de Leva, 1996)")
    
    st.divider()
    
    # Landmark indices used
    st.subheader("🔍 MediaPipe Landmark Indices Used")
    com_landmark_indices = {
        'head': [0],
        'torso': [11, 12, 23, 24],
        'left_arm': [13, 15],
        'right_arm': [14, 16],
        'left_leg': [25, 27],
        'right_leg': [26, 28],
    }
    
    landmark_df = pd.DataFrame([
        {"Segment": k.replace("_", " ").title(), "Landmark Indices": ", ".join(map(str, v))}
        for k, v in com_landmark_indices.items()
    ])
    st.dataframe(landmark_df, width="stretch", hide_index=True)
    
    st.caption("MediaPipe Pose Landmarker landmark indices (33 total landmarks)")
    
    st.divider()
    
    # Ngrok Configuration
    st.subheader("🌐 Ngrok Tunnel Configuration")
    
    ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
    ngrok_domain = os.environ.get("NGROK_DOMAIN")
    
    col1, col2 = st.columns(2)
    with col1:
        if ngrok_token:
            st.metric("Auth Token", "✅ Configured")
            st.caption(f"Token: {ngrok_token[:8]}...{ngrok_token[-4:]}")
        else:
            st.metric("Auth Token", "❌ Not Set")
            st.caption("Set NGROK_AUTH_TOKEN in .env")
    
    with col2:
        if ngrok_domain:
            st.metric("Custom Domain", "✅ Configured")
            st.caption(f"Domain: {ngrok_domain}")
        else:
            st.metric("Custom Domain", "❌ Not Set (using random)")
            st.caption("Optional: Set NGROK_DOMAIN in .env (requires paid plan)")
    
    if st.session_state.get("ngrok_url"):
        st.success(f"🔗 Active Tunnel: {st.session_state.ngrok_url}")
    
    st.divider()
    
    # Model info
    st.subheader("🤖 Model Information")
    st.markdown("""
    - **Model**: MediaPipe Pose Landmarker Heavy
    - **Input**: 1920x1080 (resized internally)
    - **Precision**: Float16
    - **Running Mode**: VIDEO (optimized for video streams)
    - **Landmarks**: 33 pose landmarks with visibility/presence scores
    """)
    
    st.divider()
    
    # Output info
    st.subheader("📁 Output Files")
    st.markdown("""
    - **output_video.mp4** - Main output with pose landmarks + COM overlay
    - **output_video_com_trail.mp4** - Full COM trajectory trail + pose overlay
    - **output_video_web.mp4** - Web-compatible H.264 re-encoded main video
    - **output_video_com_trail_web.mp4** - Web-compatible H.264 re-encoded trail video
    - **pose_annotations.json** - Complete frame-by-frame annotations
    - **output_frames/** - Individual annotated frames (JPG)
    """)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Initialize ngrok tunnel if configured
    init_ngrok_tunnel()
    
    main()