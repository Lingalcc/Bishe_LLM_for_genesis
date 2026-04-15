#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _coerce_float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except Exception:
            result.append(0.0)
    return result


def _normalize_scores(scores: list[float]) -> list[float]:
    clipped = [max(0.0, float(score)) for score in scores]
    total = sum(clipped)
    if total <= 0.0:
        return [0.0 for _ in clipped]
    return [score / total for score in clipped]


def _build_cumulative(values: list[float]) -> list[float]:
    cumulative: list[float] = []
    running = 0.0
    for value in values:
        running += float(value)
        cumulative.append(running)
    return cumulative


def _parse_layer_index(text: str) -> int | None:
    if isinstance(text, int):
        return int(text)
    if not isinstance(text, str):
        return None
    parts = text.rsplit("_", 1)
    if len(parts) != 2:
        try:
            return int(text)
        except Exception:
            return None
    try:
        return int(parts[1])
    except Exception:
        return None


def _extract_scores_from_layers_list(payload: dict[str, Any]) -> list[float]:
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return []

    indexed_scores: dict[int, float] = {}
    for layer_value in layers:
        if not isinstance(layer_value, dict):
            continue
        try:
            layer_index = int(layer_value.get("layer"))
            layer_score = float(layer_value.get("score", 0.0))
        except Exception:
            continue
        indexed_scores[layer_index] = layer_score

    if not indexed_scores:
        return []

    max_layer_index = max(indexed_scores)
    return [float(indexed_scores.get(layer_index, 0.0)) for layer_index in range(max_layer_index + 1)]


def _extract_scores_from_layers_mapping(payload: dict[str, Any]) -> list[float]:
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        return []

    indexed_scores: dict[int, float] = {}
    for layer_value in layers.values():
        if not isinstance(layer_value, dict):
            continue
        try:
            layer_index = int(layer_value.get("layer_index"))
            layer_score = float(layer_value.get("score", 0.0))
        except Exception:
            continue
        indexed_scores[layer_index] = layer_score

    if not indexed_scores:
        return []

    max_layer_index = max(indexed_scores)
    return [float(indexed_scores.get(layer_index, 0.0)) for layer_index in range(max_layer_index + 1)]


def _extract_scores_from_ranking(payload: dict[str, Any]) -> list[float]:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        return []

    indexed_scores: dict[int, float] = {}
    for item in ranking:
        if not isinstance(item, dict):
            continue
        layer_index = _parse_layer_index(str(item.get("layer", "")))
        if layer_index is None:
            continue
        try:
            indexed_scores[layer_index] = float(item.get("score", 0.0))
        except Exception:
            indexed_scores[layer_index] = 0.0

    if not indexed_scores:
        return []

    max_layer_index = max(indexed_scores)
    return [float(indexed_scores.get(layer_index, 0.0)) for layer_index in range(max_layer_index + 1)]


@dataclass(frozen=True)
class LayerImportanceProfile:
    layer_scores: list[float]
    layer_probs: list[float]
    cum_importance: list[float]
    source_format: str
    source_path: str | None = None

    @property
    def num_layers(self) -> int:
        return len(self.layer_scores)

    def score_at(self, layer_index: int) -> float:
        if layer_index < 0 or layer_index >= self.num_layers:
            return 0.0
        return float(self.layer_scores[layer_index])

    def prob_at(self, layer_index: int) -> float:
        if layer_index < 0 or layer_index >= self.num_layers:
            return 0.0
        return float(self.layer_probs[layer_index])

    def cum_importance_at(self, layer_index: int) -> float:
        if layer_index < 0:
            return 0.0
        if not self.cum_importance:
            return 0.0
        if layer_index >= len(self.cum_importance):
            return float(self.cum_importance[-1])
        return float(self.cum_importance[layer_index])

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_scores": [float(score) for score in self.layer_scores],
            "layer_probs": [float(prob) for prob in self.layer_probs],
            "cum_importance": [float(value) for value in self.cum_importance],
            "source_format": self.source_format,
            "source_path": self.source_path,
        }


def build_importance_profile(payload: dict[str, Any], *, source_path: str | None = None) -> LayerImportanceProfile:
    if not isinstance(payload, dict):
        raise TypeError("importance payload 必须是 dict。")

    layer_scores = _coerce_float_list(payload.get("layer_scores"))
    source_format = "normalized_importance"
    if not layer_scores:
        layer_scores = _extract_scores_from_layers_list(payload)
        if layer_scores:
            source_format = "legacy_layer_scores_layers_list"
    if not layer_scores:
        layer_scores = _extract_scores_from_layers_mapping(payload)
        if layer_scores:
            source_format = "legacy_layer_scores_layers"
    if not layer_scores:
        layer_scores = _extract_scores_from_ranking(payload)
        if layer_scores:
            source_format = "legacy_layer_scores_ranking"
    if not layer_scores:
        raise ValueError("无法从 importance payload 中解析 layer_scores。")

    layer_probs = _coerce_float_list(payload.get("layer_probs"))
    if len(layer_probs) != len(layer_scores):
        layer_probs = _normalize_scores(layer_scores)

    cum_importance = _coerce_float_list(payload.get("cum_importance"))
    if len(cum_importance) != len(layer_scores):
        cum_importance = _build_cumulative(layer_probs)

    return LayerImportanceProfile(
        layer_scores=[float(score) for score in layer_scores],
        layer_probs=[float(prob) for prob in layer_probs],
        cum_importance=[float(value) for value in cum_importance],
        source_format=source_format,
        source_path=source_path,
    )


def load_importance_profile(
    importance_file: str | Path,
    *,
    expected_num_layers: int | None = None,
) -> LayerImportanceProfile:
    path = Path(importance_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"importance 文件不存在: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = build_importance_profile(payload, source_path=str(path))
    if expected_num_layers is not None and profile.num_layers != int(expected_num_layers):
        raise ValueError(
            f"importance 层数与模型不一致：profile={profile.num_layers}, expected={expected_num_layers}, file={path}"
        )
    return profile


def write_importance_profile_json(profile: LayerImportanceProfile, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
