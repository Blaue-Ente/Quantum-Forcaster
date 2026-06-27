"""Модул за изтегляне и инженеринг на пазарни данни чрез yfinance.

Поддържа акции, криптовалути и форекс. Изчислява технически индикатори
и конструира етикети за насока на движението (up/down) за класификация.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from ..core.config import ENSEMBLE_CONFIG


@dataclass
class FeatureFrame:
    """Контейнер за подготвени данни и етикети."""

    features: pd.DataFrame
    labels: pd.Series
    close: pd.Series
    last_window: pd.DataFrame
    last_close: float
    symbol: str
    horizon: int


def _asset_category(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return "crypto"
    if symbol.endswith("=X"):
        return "forex"
    return "stock"


def _period_for(symbol: str) -> str:
    """По-дълга история за по-малко волатилни активи."""
    return "2y" if _asset_category(symbol) in ("stock", "forex") else "1y"


def fetch_raw(symbol: str) -> pd.DataFrame:
    """Изтегля OHLCV данни за подадения символ."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=_period_for(symbol), interval="1d")
    if df.empty:
        raise ValueError(f"Няма данни за символ '{symbol}'.")
    df = df.dropna()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавя технически индикатори към OHLCV dataframe."""
    out = df.copy()

    out["sma_10"] = SMAIndicator(out["Close"], window=10).sma_indicator()
    out["sma_30"] = SMAIndicator(out["Close"], window=30).sma_indicator()
    out["ema_12"] = EMAIndicator(out["Close"], window=12).ema_indicator()
    out["ema_26"] = EMAIndicator(out["Close"], window=26).ema_indicator()

    macd = MACD(out["Close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    out["rsi_14"] = RSIIndicator(out["Close"], window=14).rsi()

    bb = BollingerBands(out["Close"])
    out["bb_h"] = bb.bollinger_hband()
    out["bb_l"] = bb.bollinger_lband()
    out["bb_m"] = bb.bollinger_mavg()

    out["atr_14"] = AverageTrueRange(out["High"], out["Low"], out["Close"]).average_true_range()

    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_3d"] = out["Close"].pct_change(3)
    out["ret_5d"] = out["Close"].pct_change(5)
    out["vol_10"] = out["ret_1d"].rolling(10).std()

    return out


FEATURE_COLUMNS = [
    "sma_10", "sma_30", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_diff",
    "rsi_14", "bb_h", "bb_l", "bb_m", "atr_14",
    "ret_1d", "ret_3d", "ret_5d", "vol_10",
]


def build_label(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Етикет: 1 ако цената след `horizon` дни е по-висока, иначе 0."""
    future = df["Close"].shift(-horizon)
    label = (future > df["Close"]).astype(int)
    return label


def prepare_features(symbol: str, horizon: int | None = None) -> FeatureFrame:
    """Пълна подготовка на данните за даден символ."""
    horizon = horizon or ENSEMBLE_CONFIG.horizon_days

    raw = fetch_raw(symbol)
    enriched = add_indicators(raw)
    enriched["label"] = build_label(enriched, horizon)

    enriched = enriched.dropna()
    if len(enriched) < 60:
        raise ValueError(
            f"Недостатъчно наблюдения за '{symbol}': {len(enriched)} реда."
        )

    features = enriched[FEATURE_COLUMNS]
    labels = enriched["label"]
    close = enriched["Close"]

    last_window = features.tail(1).copy()
    last_close = float(close.iloc[-1])

    return FeatureFrame(
        features=features,
        labels=labels,
        close=close,
        last_window=last_window,
        last_close=last_close,
        symbol=symbol,
        horizon=horizon,
    )


def normalize_for_quantum(X: pd.DataFrame) -> np.ndarray:
    """Нормализира характеристиките в [0, π] за квантов feature map."""
    arr = X.to_numpy()
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    scaled = (arr - lo) / rng
    return scaled * np.pi


def train_test_split_temporal(
    features: pd.DataFrame, labels: pd.Series, test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Хронологичен split – без data leakage."""
    n = len(features)
    cut = int(n * (1 - test_ratio))
    X_train = features.iloc[:cut].to_numpy()
    X_test = features.iloc[cut:].to_numpy()
    y_train = labels.iloc[:cut].to_numpy()
    y_test = labels.iloc[cut:].to_numpy()
    return X_train, X_test, y_train, y_test
