"""
Ngrok tunnel utility for exposing local Streamlit app to the internet.
Run this script in a separate terminal to create a public tunnel.
"""

import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional


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


def start_ngrok_tunnel(port: int = 8501, auth_token: Optional[str] = None, domain: Optional[str] = None) -> Optional[str]:
    """
    Start ngrok tunnel for the given port.
    
    Args:
        port: Local port to tunnel (default 8501 for Streamlit)
        auth_token: Ngrok auth token (can also be set via NGROK_AUTH_TOKEN env var)
        domain: Custom domain for ngrok (requires paid plan, can also be set via NGROK_DOMAIN env var)
    
    Returns:
        Public URL if successful, None otherwise
    """
    load_env()
    
    token = auth_token or os.environ.get("NGROK_AUTH_TOKEN")
    if not token:
        print("❌ No ngrok auth token found. Set NGROK_AUTH_TOKEN in .env file")
        print("   Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        return None
    
    # Get custom domain from env if not provided
    custom_domain = domain or os.environ.get("NGROK_DOMAIN")
    
    # Check if ngrok is installed
    try:
        subprocess.run(["ngrok", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ngrok not installed. Install it first:")
        print("   macOS: brew install ngrok")
        print("   Linux: snap install ngrok")
        print("   Or download from: https://ngrok.com/download")
        return None
    
    # Configure ngrok with auth token
    try:
        subprocess.run(["ngrok", "config", "add-authtoken", token],
                       check=True, capture_output=True)
        print("✅ Ngrok auth token configured")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to configure ngrok auth: {e.stderr.decode()}")
        return None
    
    # Start tunnel
    try:
        print(f"🚀 Starting ngrok tunnel on port {port}...")
        if custom_domain:
            print(f"   Using custom domain: {custom_domain}")
        
        # Build ngrok command
        ngrok_cmd = ["ngrok", "http", str(port), "--log=stdout"]
        if custom_domain:
            ngrok_cmd.extend(["--domain", custom_domain])
        
        # Start ngrok in background
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
                    public_url = tunnel["public_url"]
                    print(f"\n✅ Ngrok tunnel started successfully!")
                    print(f"   Local:  http://localhost:{port}")
                    print(f"   Public: {public_url}")
                    if custom_domain:
                        print(f"   Custom Domain: {custom_domain}")
                    print(f"   Web UI: http://localhost:4040")
                    print(f"\n📋 Share this URL: {public_url}")
                    print(f"   Press Ctrl+C to stop tunnel\n")
                    
                    # Keep running until interrupted
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping tunnel...")
                        process.terminate()
                    return public_url
        except requests.RequestException as e:
            print(f"❌ Could not get tunnel URL from ngrok API: {e}")
            process.terminate()
            return None
        
    except Exception as e:
        print(f"❌ Failed to start ngrok: {e}")
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


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    start_ngrok_tunnel(port)