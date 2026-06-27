"""Vercel ASGI handler for FastAPI application."""
import sys
from pathlib import Path

# Add the quantum-financial-forecaster directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "quantum-financial-forecaster"
sys.path.insert(0, str(BACKEND_DIR))

# Import and expose the FastAPI app
from api.server import app

# Export for Vercel
handler = app
