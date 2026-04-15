#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.eval_core.importance_loader import LayerImportanceProfile


def parse_exit_layers(raw_layers: Any) -> list[int]:
    if raw_layers is None:
        return []
    if isinstance(raw_layers, str):
        values = [part.strip() for part in raw_layers.split(",") if part.strip()]
    elif isinstance(raw_layers, (list, tuple, set)):
        values = list(raw_layers)
    else:
        values = [raw_layers]

    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except Exception:
            continue
    return sorted(set(parsed))


@dataclass(frozen=True)
class LayerProbeResult:
    layer_index: int
    is_candidate: bool
    max_prob: float
    cum_importance: float
    meets_importance: bool
    meets_confidence: bool
    should_exit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "is_candidate": self.is_candidate,
            "max_prob": float(self.max_prob),
            "cum_importance": float(self.cum_importance),
            "meets_importance": self.meets_importance,
            "meets_confidence": self.meets_confidence,
            "should_exit": self.should_exit,
        }


class EarlyExitPolicy:
    """统一阈值版 IIDEE 退出策略。"""

    def __init__(
        self,
        *,
        exit_layers: list[int],
        importance_profile: LayerImportanceProfile,
        tau_importance: float,
        tau_confidence: float,
        total_layers: int,
    ) -> None:
        if total_layers <= 0:
            raise ValueError("total_layers 必须大于 0。")
        if not 0.0 <= float(tau_importance) <= 1.0:
            raise ValueError("tau_importance 必须在 [0, 1] 范围内。")
        if not 0.0 <= float(tau_confidence) <= 1.0:
            raise ValueError("tau_confidence 必须在 [0, 1] 范围内。")

        self.total_layers = int(total_layers)
        self.exit_layers = [layer for layer in sorted(set(exit_layers)) if 0 <= int(layer) < self.total_layers]
        if not self.exit_layers:
            raise ValueError("exit_layers 不能为空，且必须落在合法层号范围内。")

        self.importance_profile = importance_profile
        self.tau_importance = float(tau_importance)
        self.tau_confidence = float(tau_confidence)

    def effective_exit_layers(self, *, layer_cap: int | None = None) -> list[int]:
        if layer_cap is None:
            return list(self.exit_layers)
        return [layer for layer in self.exit_layers if layer <= int(layer_cap)]

    def probe(self, *, layer_index: int, max_prob: float) -> LayerProbeResult:
        layer_index = int(layer_index)
        max_prob = float(max_prob)
        cum_importance = float(self.importance_profile.cum_importance_at(layer_index))
        is_candidate = layer_index in set(self.exit_layers)
        meets_importance = cum_importance >= self.tau_importance
        meets_confidence = max_prob >= self.tau_confidence
        return LayerProbeResult(
            layer_index=layer_index,
            is_candidate=is_candidate,
            max_prob=max_prob,
            cum_importance=cum_importance,
            meets_importance=meets_importance,
            meets_confidence=meets_confidence,
            should_exit=bool(is_candidate and meets_importance and meets_confidence),
        )
