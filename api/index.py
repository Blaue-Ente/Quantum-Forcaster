"""Vercel ASGI handler for FastAPI application."""
import sys
from pathlib import Path

# First try to use the full-featured server with quantum capabilities
# If that fails (heavy dependencies), fall back to minimal server
try:
    BACKEND_DIR = Path(__file__).resolve().parent.parent / "quantum-financial-forecaster"
    sys.path.insert(0, str(BACKEND_DIR))
    from api.server import app
except Exception as e:
    print(f"[v0] Failed to import full server: {e}")
    print("[v0] Falling back to minimal server")
    # Use minimal server without heavy ML dependencies
    from api.server_minimal import app

# Export for Vercel
handler = app
