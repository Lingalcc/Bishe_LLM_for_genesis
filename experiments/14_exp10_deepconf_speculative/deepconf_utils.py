#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from typing import Any


def aggregate_token_confidence(
    *,
    token_confidences: list[float],
    actual_token_probs: list[float],
    bottom_fraction: float,
    tail_fraction: float,
    avg_weight: float,
    bottom_weight: float,
    tail_weight: float,
    actual_prob_weight: float,
) -> dict[str, float]:
    if not token_confidences:
        return {
            "num_generated_tokens": 0.0,
            "avg_confidence": 0.0,
            "bottom_confidence": 0.0,
            "tail_confidence": 0.0,
            "avg_actual_token_prob": 0.0,
            "deepconf_score": 0.0,
        }

    count = len(token_confidences)
    bottom_count = max(1, int(math.ceil(count * max(0.0, min(1.0, bottom_fraction)))))
    tail_count = max(1, int(math.ceil(count * max(0.0, min(1.0, tail_fraction)))))

    avg_confidence = float(sum(token_confidences) / count)
    bottom_confidence = float(sum(sorted(token_confidences)[:bottom_count]) / bottom_count)
    tail_confidence = float(sum(token_confidences[-tail_count:]) / tail_count)
    avg_actual_token_prob = float(sum(actual_token_probs) / len(actual_token_probs)) if actual_token_probs else 0.0

    deepconf_score = (
        avg_weight * avg_confidence
        + bottom_weight * bottom_confidence
        + tail_weight * tail_confidence
        + actual_prob_weight * avg_actual_token_prob
    )
    return {
        "num_generated_tokens": float(count),
        "avg_confidence": avg_confidence,
        "bottom_confidence": bottom_confidence,
        "tail_confidence": tail_confidence,
        "avg_actual_token_prob": avg_actual_token_prob,
        "deepconf_score": float(deepconf_score),
    }


def select_weighted_confidence_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates 不能为空。")

    parse_ok_candidates = [item for item in candidates if bool(item.get("parse_ok", False))]
    if not parse_ok_candidates:
        return max(
            candidates,
            key=lambda item: (
                float(item.get("deepconf_score", 0.0)),
                float(item.get("avg_actual_token_prob", 0.0)),
                -int(item.get("candidate_index", 0)),
            ),
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in parse_ok_candidates:
        key = str(candidate.get("canonical_commands", ""))
        grouped.setdefault(key, []).append(candidate)

    def group_rank(items: list[dict[str, Any]]) -> tuple[float, int, float]:
        total_score = float(sum(float(item.get("deepconf_score", 0.0)) for item in items))
        best_score = float(max(float(item.get("deepconf_score", 0.0)) for item in items))
        return total_score, len(items), best_score

    best_group = max(grouped.values(), key=group_rank)
    return max(
        best_group,
        key=lambda item: (
            float(item.get("deepconf_score", 0.0)),
            float(item.get("avg_actual_token_prob", 0.0)),
            -int(item.get("candidate_index", 0)),
        ),
    )


def summarize_candidate_pool(candidates: list[dict[str, Any]]) -> dict[str, float]:
    if not candidates:
        return {
            "candidate_count": 0.0,
            "candidate_parse_ok_rate": 0.0,
            "candidate_avg_deepconf_score": 0.0,
            "candidate_best_deepconf_score": 0.0,
            "candidate_avg_generated_tokens": 0.0,
            "candidate_unique_prediction_count": 0.0,
            "candidate_unique_action_signature_count": 0.0,
            "candidate_duplicate_prediction_rate": 0.0,
            "candidate_duplicate_action_signature_rate": 0.0,
            "candidate_total_resample_attempts": 0.0,
        }

    unique_predictions: set[str] = set()
    unique_action_signatures: set[str] = set()
    parse_ok_count = sum(1 for item in candidates if bool(item.get("parse_ok", False)))
    scores = [float(item.get("deepconf_score", 0.0)) for item in candidates]
    generated_tokens = [float(item.get("generated_tokens", 0.0)) for item in candidates]
    total_resample_attempts = sum(int(item.get("resample_attempts", 0)) for item in candidates)

    for item in candidates:
        commands = item.get("canonical_commands")
        prediction = item.get("prediction")
        action_signature = item.get("action_signature")
        if commands is not None:
            unique_predictions.add(str(commands))
        elif isinstance(prediction, str):
            unique_predictions.add(" ".join(prediction.split()))
        else:
            unique_predictions.add(json.dumps(prediction, ensure_ascii=False, sort_keys=True))

        if action_signature is None:
            unique_action_signatures.add("<none>")
        elif isinstance(action_signature, list):
            unique_action_signatures.add(json.dumps(action_signature, ensure_ascii=False))
        else:
            unique_action_signatures.add(str(action_signature))

    return {
        "candidate_count": float(len(candidates)),
        "candidate_parse_ok_rate": float(parse_ok_count / len(candidates)),
        "candidate_avg_deepconf_score": float(sum(scores) / len(scores)),
        "candidate_best_deepconf_score": float(max(scores)),
        "candidate_avg_generated_tokens": float(sum(generated_tokens) / len(generated_tokens)),
        "candidate_unique_prediction_count": float(len(unique_predictions)),
        "candidate_unique_action_signature_count": float(len(unique_action_signatures)),
        "candidate_duplicate_prediction_rate": float(1.0 - (len(unique_predictions) / len(candidates))),
        "candidate_duplicate_action_signature_rate": float(1.0 - (len(unique_action_signatures) / len(candidates))),
        "candidate_total_resample_attempts": float(total_resample_attempts),
    }
