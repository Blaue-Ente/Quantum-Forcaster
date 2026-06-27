"""Конфигурация на адаптивния квантов прогнозен генератор."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class QuantumConfig:
    """Параметри на квантовия класификатор (VQC)."""

    feature_map_reps: int = 2
    ansatz_reps: int = 2
    entanglement: str = "circular"
    max_qubits: int = 8
    training_samples: int = 200
    shots: int = 1024


@dataclass(frozen=True)
class EnsembleConfig:
    """Параметри на хибридния ансамбъл."""

    classical_weight: float = 0.55
    quantum_weight: float = 0.45
    horizon_days: int = 7
    confidence_threshold: float = 0.60


@dataclass(frozen=True)
class AssetUniverse:
    """Поддържани активи по категории."""

    stocks: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    )
    crypto: List[str] = field(
        default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    )
    forex: List[str] = field(
        default_factory=lambda: ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    )

    def all_symbols(self) -> List[str]:
        return [*self.stocks, *self.crypto, *self.forex]


# Единни инстанции за целия проект
QUANTUM_CONFIG = QuantumConfig()
ENSEMBLE_CONFIG = EnsembleConfig()
ASSET_UNIVERSE = AssetUniverse()
