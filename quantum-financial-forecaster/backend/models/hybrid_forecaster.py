"""Хибриден ансамбъл: адаптивен квантов VQC + класическо ML.

Ансамбълът комбинира:
  * AdaptiveQuantumClassifier (VQC с ZZFeatureMap + RealAmplitudes);
  * RandomForestClassifier + GradientBoostingClassifier (класическо ML).

Тежестите на ансамбъла се пренастройват адаптивно според волатилността на
актива: при висока волатилност класическите модели получават повече тегло
(по-стабилни при шум), при ниска волатилност квантовият модел има по-голяма
тежест (по-добра експресивност при гладки патерни).

Включва също детекция на пазарен режим (trending/mean-reverting/volatile)
и прости 价格 projections за хоризонта на прогнозата.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..core.config import ENSEMBLE_CONFIG, EnsembleConfig
from .data_loader import FeatureFrame, prepare_features, train_test_split_temporal
from .quantum_classifier import AdaptiveQuantumClassifier, QuantumPrediction


@dataclass
class RegimeInfo:
    """Информация за пазарния режим."""

    regime: str
    volatility_score: float
    trend_score: float


@dataclass
class ClassicalResult:
    """Резултат от класическия ансамбъл."""

    probability_up: float
    confidence: float
    rf_proba: float
    gb_proba: float


@dataclass
class ForecastResult:
    """Пълна прогноза за даден символ."""

    symbol: str
    horizon_days: int
    direction: str
    probability_up: float
    confidence: float
    last_close: float
    projected_close: float
    projected_change_pct: float
    regime: RegimeInfo
    quantum: QuantumPrediction
    classical: ClassicalResult
    weights: Dict[str, float]
    history: List[Dict[str, float]] = field(default_factory=list)
    quantum_accuracy: float = 0.0
    classical_accuracy: float = 0.0


def detect_regime(features: pd.DataFrame) -> RegimeInfo:
    """Проста детекция на пазарен режим."""
    ret = features["ret_1d"].tail(30)
    vol = float(ret.std())
    trend = float(ret.sum())

    vol_score = float(np.clip(vol * 10.0, 0.0, 1.0))
    trend_score = float(np.clip(abs(trend) / (vol + 1e-8) / 10.0, 0.0, 1.0))

    if vol_score > 0.6:
        regime = "volatile"
    elif trend_score > 0.5:
        regime = "trending"
    else:
        regime = "mean-reverting"

    return RegimeInfo(
        regime=regime,
        volatility_score=vol_score,
        trend_score=trend_score,
    )


def adaptive_weights(regime: RegimeInfo, base: EnsembleConfig) -> Dict[str, float]:
    """Адаптира тежестите според режима."""
    q = base.quantum_weight
    c = base.classical_weight
    if regime.regime == "volatile":
        q *= 0.7
        c *= 1.3
    elif regime.regime == "trending":
        q *= 1.2
        c *= 0.8
    total = q + c
    return {"quantum": q / total, "classical": c / total}


class HybridForecaster:
    """Хибриден адаптивен прогнозен генератор."""

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or ENSEMBLE_CONFIG
        self._scaler = StandardScaler()
        self._rf: RandomForestClassifier | None = None
        self._gb: GradientBoostingClassifier | None = None
        self._quantum = AdaptiveQuantumClassifier()
        self._quantum_accuracy: float = 0.0
        self._classical_accuracy: float = 0.0

    def fit(self, frame: FeatureFrame) -> "HybridForecaster":
        X_train, X_test, y_train, y_test = train_test_split_temporal(
            frame.features, frame.labels, test_ratio=0.2
        )

        # Класически модели
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)

        self._rf = RandomForestClassifier(
            n_estimators=120, max_depth=6, random_state=42, n_jobs=-1
        )
        self._gb = GradientBoostingClassifier(
            n_estimators=120, max_depth=3, random_state=42
        )
        self._rf.fit(X_train_s, y_train)
        self._gb.fit(X_train_s, y_train)

        self._classical_accuracy = float(
            0.5 * self._rf.score(X_test_s, y_test)
            + 0.5 * self._gb.score(X_test_s, y_test)
        )

        # Квантов модел (субсемплинг за скорост)
        n_train = min(self._quantum.config.training_samples, X_train.shape[0])
        idx = np.linspace(0, X_train.shape[0] - 1, n_train).astype(int)
        Xq = X_train[idx]
        yq = y_train[idx]
        self._quantum.fit(Xq, yq)
        self._quantum_accuracy = self._quantum.evaluate(X_test, y_test)

        return self

    def predict(self, frame: FeatureFrame) -> ForecastResult:
        if self._rf is None or self._gb is None:
            raise RuntimeError("Моделът не е обучен. Извикайте fit() първо.")

        regime = detect_regime(frame.features)
        weights = adaptive_weights(regime, self.config)

        # Последен вектор
        last = frame.last_window.iloc[0].to_numpy()
        last_scaled = self._scaler.transform(last.reshape(1, -1))[0]

        # Квантова прогноза
        q_pred = self._quantum.predict_single(last)

        # Класическа прогноза
        rf_proba = float(self._rf.predict_proba(last_scaled.reshape(1, -1))[0][1])
        gb_proba = float(self._gb.predict_proba(last_scaled.reshape(1, -1))[0][1])
        c_proba = 0.5 * rf_proba + 0.5 * gb_proba
        c_conf = float(abs(c_proba - 0.5) * 2.0)

        classical = ClassicalResult(
            probability_up=c_proba,
            confidence=c_conf,
            rf_proba=rf_proba,
            gb_proba=gb_proba,
        )

        # Ансамбъл
        p_up = weights["quantum"] * q_pred.probability_up + weights["classical"] * c_proba
        direction = "UP" if p_up >= 0.5 else "DOWN"
        confidence = float(abs(p_up - 0.5) * 2.0)

        # Проекция на цена (опростена)
        ret_5d = float(frame.last_window["ret_5d"].iloc[0])
        projected_change = (p_up - 0.5) * 2 * 0.05 + ret_5d * 0.3
        projected_change = float(np.clip(projected_change, -0.15, 0.15))
        projected_close = frame.last_close * (1.0 + projected_change)

        history = [
            {"date": d.strftime("%Y-%m-%d"), "close": float(c)}
            for d, c in frame.close.tail(60).items()
        ]

        return ForecastResult(
            symbol=frame.symbol,
            horizon_days=frame.horizon,
            direction=direction,
            probability_up=float(p_up),
            confidence=confidence,
            last_close=frame.last_close,
            projected_close=float(projected_close),
            projected_change_pct=float(projected_change * 100),
            regime=regime,
            quantum=q_pred,
            classical=classical,
            weights=weights,
            history=history,
            quantum_accuracy=self._quantum_accuracy,
            classical_accuracy=self._classical_accuracy,
        )


def forecast_symbol(symbol: str, horizon: int | None = None) -> ForecastResult:
    """Генерира пълна прогноза за даден символ."""
    frame = prepare_features(symbol, horizon=horizon)
    forecaster = HybridForecaster()
    forecaster.fit(frame)
    return forecaster.predict(frame)
