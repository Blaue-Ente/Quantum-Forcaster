"""FastAPI сървър за адаптивния квантов прогнозен генератор - Vercel wrapper."""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

# Add the quantum-financial-forecaster directory to sys.path for imports
BACKEND_DIR = Path(__file__).resolve().parent.parent / "quantum-financial-forecaster"
sys.path.insert(0, str(BACKEND_DIR))

from backend.core.config import ASSET_UNIVERSE
from backend.models.hybrid_forecaster import forecast_symbol

BASE_DIR = BACKEND_DIR
STATIC_DIR = BASE_DIR / "frontend" / "static"
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"

app = FastAPI(
    title="Adaptive Quantum Financial Forecaster",
    description=(
        "Хибриден адаптивен квантов алгоритъмичен генератор на прогнози "
        "за финансовите пазари (акции, крипто, форекс)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _to_dict(obj) -> Dict:
    """Рекурсивно конвертира dataclass към dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    return obj


@app.get("/")
async def index():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/api/assets")
async def list_assets() -> Dict[str, List[str]]:
    """Връща поддържаните активи по категории."""
    return {
        "stocks": ASSET_UNIVERSE.stocks,
        "crypto": ASSET_UNIVERSE.crypto,
        "forex": ASSET_UNIVERSE.forex,
    }


@app.get("/api/forecast")
async def get_forecast(
    symbol: str = Query(..., description="Тикер (напр. AAPL, BTC-USD, EURUSD=X)"),
    horizon: int = Query(7, ge=1, le=30, description="Хоризонт в дни"),
):
    """Генерира хибридна квантово-класическа прогноза."""
    try:
        result = forecast_symbol(symbol, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Вътрешна грешка: {exc}")

    return _to_dict(result)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "quantum-financial-forecaster"}
