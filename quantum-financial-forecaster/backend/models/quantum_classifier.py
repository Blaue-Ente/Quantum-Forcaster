"""Адаптивен квантов класификатор, базиран на Variational Quantum Circuit.

Използва ZZFeatureMap за кодиране на характеристики в квантово състояние и
RealAmplitudes ansatz за параметризирана квантова верига. VQC се обучава
чрез COBYLA оптимизатор върху симулатор (Aer qasm_simulator).

Алгоритъмът е "адаптивен" в смисъла, че:
  * избира брой кубита според броя характеристики (до max_qubits);
  * пренастройва feature map reps спрямо волатилността на актива;
  * при нестабилни режими използва по-дълбок ansatz за по-добра експресивност.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Qiskit импорти
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit import Aer

from ..core.config import QUANTUM_CONFIG, QuantumConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class QuantumPrediction:
    """Резултат от квантовия класификатор."""

    label: int
    probability_up: float
    confidence: float
    n_qubits: int
    feature_map_depth: int
    ansatz_depth: int


def _volatility_regime(X: np.ndarray) -> float:
    """Оценка на волатилност (за адаптивност)."""
    if X.shape[0] < 2:
        return 0.5
    last = X[-1]
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    z = np.abs((last - mean) / std)
    return float(np.clip(np.mean(z) / 3.0, 0.0, 1.0))


def _select_n_qubits(n_features: int, cfg: QuantumConfig) -> int:
    """Адаптира броя кубита към характеристиките."""
    return max(2, min(cfg.max_qubits, int(np.ceil(np.log2(max(n_features, 2))))))


def _adaptive_depth(volatility: float, base_reps: int) -> int:
    """По-дълбок ansatz при висока волатилност."""
    if volatility > 0.6:
        return base_reps + 1
    if volatility < 0.25:
        return max(1, base_reps - 1)
    return base_reps


def build_quantum_circuit(
    n_qubits: int,
    n_features: int,
    feature_reps: int,
    ansatz_reps: int,
    entanglement: str,
) -> Tuple[ZZFeatureMap, RealAmplitudes]:
    """Конструира feature map и ansatz."""
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=feature_reps,
        entanglement=entanglement,
    )
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=ansatz_reps,
        entanglement=entanglement,
    )
    return feature_map, ansatz


class AdaptiveQuantumClassifier:
    """VQC обвивка с адаптивни параметри."""

    def __init__(self, config: QuantumConfig | None = None) -> None:
        self.config = config or QUANTUM_CONFIG
        self._vqc: VQC | None = None
        self._n_qubits: int = 0
        self._feature_reps: int = self.config.feature_map_reps
        self._ansatz_reps: int = self.config.ansatz_reps
        self._backend = Aer.get_backend("qasm_simulator")
        self._quantum_instance = QuantumInstance(
            self._backend, shots=self.config.shots, seed_simulator=42, seed_transpiler=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveQuantumClassifier":
        """Обучава VQC върху подадените данни."""
        vol = _volatility_regime(X)
        self._n_qubits = _select_n_qubits(X.shape[1], self.config)
        self._ansatz_reps = _adaptive_depth(vol, self.config.ansatz_reps)

        # PCA-подобна селекция: първите n_qubits характеристики
        X_reduced = X[:, : self._n_qubits]
        # Нормализираме в [0, π] за feature map
        X_norm = self._scale_to_pi(X_reduced)

        feature_map, ansatz = build_quantum_circuit(
            n_qubits=self._n_qubits,
            n_features=self._n_qubits,
            feature_reps=self._feature_reps,
            ansatz_reps=self._ansatz_reps,
            entanglement=self.config.entanglement,
        )

        optimizer = COBYLA(maxiter=80)

        self._vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=self._quantum_instance,
        )

        self._vqc.fit(X_norm, y)
        return self

    def predict_single(self, x_row: np.ndarray) -> QuantumPrediction:
        """Прогноза за един сэмпл."""
        if self._vqc is None:
            raise RuntimeError("VQC не е обучен. Извикайте fit() първо.")

        x_reduced = x_row[: self._n_qubits].reshape(1, -1)
        x_norm = self._scale_to_pi(x_reduced)

        proba = self._vqc.predict_proba(x_norm)[0]
        # Клас 1 = "up"
        p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
        label = int(p_up >= 0.5)
        confidence = float(abs(p_up - 0.5) * 2.0)

        return QuantumPrediction(
            label=label,
            probability_up=p_up,
            confidence=confidence,
            n_qubits=self._n_qubits,
            feature_map_depth=self._feature_reps,
            ansatz_depth=self._ansatz_reps,
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Точност на тестови данни."""
        if self._vqc is None:
            raise RuntimeError("VQC не е обучен.")
        X_reduced = X[:, : self._n_qubits]
        X_norm = self._scale_to_pi(X_reduced)
        preds = self._vqc.predict(X_norm)
        return float(np.mean(preds == y))

    @staticmethod
    def _scale_to_pi(arr: np.ndarray) -> np.ndarray:
        lo = arr.min(axis=0, keepdims=True)
        hi = arr.max(axis=0, keepdims=True)
        rng = np.where(hi - lo == 0, 1.0, hi - lo)
        scaled = (arr - lo) / rng
        return scaled * np.pi
