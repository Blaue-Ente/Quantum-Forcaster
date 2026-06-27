"""Minimal FastAPI server for Vercel - no heavy dependencies."""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Dict, List
import json

app = FastAPI(
    title="Adaptive Quantum Financial Forecaster",
    description="Quantum-Classical ML Financial Forecaster",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "quantum-financial-forecaster" / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "quantum-financial-forecaster" / "frontend" / "static"

# Mock data for supported assets
ASSET_UNIVERSE = {
    "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "crypto": ["BTC-USD", "ETH-USD", "XRP-USD"],
    "forex": ["EURUSD=X", "GBPUSD=X", "JPYUSD=X"]
}


@app.get("/")
async def index():
    """Serve the main HTML page."""
    if TEMPLATES_DIR.exists():
        return FileResponse(TEMPLATES_DIR / "index.html")
    return {"message": "Quantum Financial Forecaster API"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "quantum-financial-forecaster"}


@app.get("/api/assets")
async def list_assets() -> Dict[str, List[str]]:
    """List supported assets."""
    return ASSET_UNIVERSE


@app.get("/api/forecast")
async def get_forecast(
    symbol: str = Query(..., description="Ticker symbol"),
    horizon: int = Query(7, ge=1, le=30, description="Forecast horizon in days"),
):
    """Get a forecast for a symbol."""
    # Validate symbol
    all_symbols = (
        ASSET_UNIVERSE.get("stocks", [])
        + ASSET_UNIVERSE.get("crypto", [])
        + ASSET_UNIVERSE.get("forex", [])
    )
    
    if symbol not in all_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol {symbol} not supported. Try one of: {', '.join(all_symbols[:5])}..."
        )
    
    # Return mock forecast
    return {
        "symbol": symbol,
        "horizon_days": horizon,
        "direction": "up",
        "probability_up": 0.65,
        "confidence": 0.58,
        "last_close": 150.25,
        "projected_close": 155.50,
        "projected_change_pct": 3.49,
        "regime": {
            "regime": "trending",
            "volatility_score": 0.42,
            "trend_score": 0.65
        },
        "quantum": {
            "prediction": "up",
            "probability": 0.68,
            "circuit_depth": 15,
            "eigenvalue": 0.72
        },
        "classical": {
            "probability_up": 0.62,
            "confidence": 0.54,
            "rf_proba": 0.60,
            "gb_proba": 0.64
        },
        "weights": {
            "quantum": 0.4,
            "random_forest": 0.3,
            "gradient_boosting": 0.3
        },
        "quantum_accuracy": 0.71,
        "classical_accuracy": 0.68
    }


# Mount static files if they exist
if STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
