# 🧗 **BoulderSense** — AI-Powered Climbing Motion Analysis

> **Transform your climbing videos into biomechanical insights.** Upload a bouldering clip and get real-time pose detection, Center of Mass trajectory analysis, stability metrics, and interactive visualizations — all in a sleek Streamlit dashboard.

**BoulderSense** is a Streamlit-based web application for analyzing bouldering/climbing videos using MediaPipe Pose Landmarker. Upload a video to get real-time pose detection, Center of Mass (COM) calculation with EMA smoothing, and comprehensive analytics.

## Demo

Demo Gif:  <img width="1080" height="618" alt="demo" src="https://github.com/user-attachments/assets/99a28539-2786-4518-ae9e-3246a42daa54" />

**Demo source:** [Janja Garnbret On All Boulders — IFSC Climbing World Championships Seoul 2025 Semi-Finals & Finals](https://www.youtube.com/watch?v=vjI2dzZiF-k), by [Sharing The Art](https://www.youtube.com/@sharingtheart).



## Features

- 🎬 **Video Upload** - Support for MP4, MOV, AVI, MKV formats (up to 500MB)
- 🔄 **Real-time Processing** - Background processing with live progress updates
- 📹 **Progressive Preview** - 5-second segment previews during processing
- 📊 **Dual Output Videos**:
  - Main: Pose landmarks + raw/smoothed COM overlay
  - Trail: Full COM trajectory curve + pose overlay
- 🌐 **Web-Compatible Output** - Automatic H.264 re-encoding for browser playback
- 📈 **Interactive Analytics** - Plotly charts for COM trajectory, displacement, timing
- 📄 **JSON/CSV Export** - Full frame-by-frame annotations
- ⚙️ **Configurable Settings** - EMA smoothing, segment weights, output options
- 🌐 **Ngrok Tunnel Integration** - Built-in public tunnel support for remote access
- 📱 **Four-Tab Interface** - Live Preview, Results, Analytics, Configuration

## Installation

### Option 1: Local Python Environment (ngrok not included)

```bash
# Clone or navigate to the project directory
cd mediapipe-bouldering-pose

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required for video segment splitting and web re-encoding)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# Windows:
# Download from https://ffmpeg.org/download.html
```


### Option 2: Docker (Recommended, includes ngrok tunnel support)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run directly with Docker
docker build -t bouldering-pose .
docker run -p 8501:8501 -v $(pwd)/temp_uploads:/app/temp_uploads -v $(pwd)/temp_outputs:/app/temp_outputs bouldering-pose
```

The app will be available at `http://localhost:8501`

#### With Ngrok Tunnel (Docker Compose)
```bash
# Copy .env.example to .env and add your ngrok auth token
cp .env.example .env
# Edit .env and add: NGROK_AUTH_TOKEN=your_token_here

# Start with ngrok (uncomment ngrok service in docker-compose.yml first)
docker-compose --profile tunnel up --build
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Workflow

1. **Upload** a climbing/bouldering video using the sidebar
2. **Configure** processing settings (EMA alpha, output options)
3. **Click "Start Processing"** - processing runs in background
4. **Watch Live Preview** - 5-second segments appear as they're generated
5. **View Results** - Switch to Results tab for final videos and statistics
6. **Analyze** - Check Analytics tab for interactive charts
7. **Download** - Export videos, JSON annotations, or CSV data

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| EMA Alpha | 0.3 | Smoothing factor (0.1-0.5). Lower = smoother |
| Save Frames | ✅ | Save individual annotated frames as JPGs |
| Save JSON | ✅ | Save full frame-by-frame annotations |
| Save Trail Video | ✅ | Generate full COM trajectory video |
| Segment Duration | 5s | Length of preview segments |
| Ngrok Auth Token | - | Set in .env for public tunnel |
| Ngrok Domain | - | Optional custom domain (paid plan) |


## JSON Annotation Format

```json
{
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "total_frames": 900,
    "processed_frames": 900
  },
  "com_settings": {
    "ema_alpha": 0.3,
    "segment_weights": {...}
  },
  "timing_stats": {...},
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "com_raw": {"x": 960.5, "y": 540.2},
      "com_smoothed": {"x": 960.5, "y": 540.2},
      "landmarks": [
        {"index": 0, "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99, "presence": 0.98},
        ...
      ]
    },
    ...
  ]
}
```

## COM Calculation

The Center of Mass is calculated using biomechanical segment weights:

| Segment | Weight | Landmarks |
|---------|--------|-----------|
| Head | 7% | Nose (0) |
| Torso | 43% | Shoulders (11,12), Hips (23,24) |
| Left Arm | 5% | Elbow (13), Wrist (15) |
| Right Arm | 5% | Elbow (14), Wrist (16) |
| Left Leg | 16% | Knee (25), Ankle (27) |
| Right Leg | 16% | Knee (26), Ankle (28) |

EMA smoothing: `smoothed = α × raw + (1-α) × previous_smoothed`

## License

This project uses MediaPipe (Apache 2.0) and OpenCV (Apache 2.0).

## Acknowledgments

- [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
