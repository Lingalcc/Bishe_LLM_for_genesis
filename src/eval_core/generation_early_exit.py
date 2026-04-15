#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from src.eval_core.early_exit_policy import EarlyExitPolicy


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _update_string_state(
    text_fragment: str,
    *,
    in_open_string: bool,
    pending_escape: bool,
) -> tuple[bool, bool]:
    if not text_fragment:
        return in_open_string, pending_escape

    for ch in text_fragment:
        if pending_escape:
            pending_escape = False
            continue
        if ch == "\\" and in_open_string:
            pending_escape = True
            continue
        if ch == '"':
            in_open_string = not in_open_string
    return in_open_string, pending_escape


def summarize_token_exit_traces(token_traces: list[dict[str, Any]], *, total_layers: int) -> dict[str, Any]:
    if total_layers <= 0:
        raise ValueError("total_layers 必须大于 0。")

    if not token_traces:
        return {
            "token_count": 0,
            "avg_exit_layer": float(total_layers),
            "avg_exit_ratio": 1.0,
            "exit_triggered_tokens": 0,
            "forced_cap_tokens": 0,
            "max_prob_mean": 0.0,
            "token_exit_layer_histogram": {},
        }

    exit_layers = [int(item.get("exit_layer", total_layers - 1)) for item in token_traces]
    max_probs = [float(item.get("max_prob", 0.0)) for item in token_traces]
    histogram = Counter(exit_layers)
    exit_triggered_tokens = sum(1 for item in token_traces if bool(item.get("exit_triggered", False)))
    forced_cap_tokens = sum(1 for item in token_traces if bool(item.get("forced_by_cache_cap", False)))
    string_guard_blocked_tokens = sum(1 for item in token_traces if bool(item.get("string_guard_blocked_exit", False)))
    draft_only_candidate_tokens = sum(1 for item in token_traces if bool(item.get("draft_only_candidate", False)))
    draft_verified_matches = sum(1 for item in token_traces if bool(item.get("draft_verified_match", False)))
    draft_verified_mismatches = sum(1 for item in token_traces if bool(item.get("draft_verified_mismatch", False)))
    avg_exit_layer = _mean([float(layer) for layer in exit_layers])
    candidate_probe_stats: dict[int, dict[str, float]] = {}
    for token_trace in token_traces:
        probe_details = token_trace.get("probe_details", [])
        if not isinstance(probe_details, list):
            continue
        for probe in probe_details:
            if not isinstance(probe, dict):
                continue
            layer_index = int(probe.get("layer_index", -1))
            if layer_index < 0:
                continue
            stat = candidate_probe_stats.setdefault(
                layer_index,
                {
                    "probe_count": 0.0,
                    "max_prob_sum": 0.0,
                    "meets_importance_count": 0.0,
                    "meets_confidence_count": 0.0,
                    "exit_count": 0.0,
                },
            )
            stat["probe_count"] += 1.0
            stat["max_prob_sum"] += float(probe.get("max_prob", 0.0))
            stat["meets_importance_count"] += 1.0 if bool(probe.get("meets_importance", False)) else 0.0
            stat["meets_confidence_count"] += 1.0 if bool(probe.get("meets_confidence", False)) else 0.0
            stat["exit_count"] += 1.0 if bool(probe.get("should_exit", False)) else 0.0

    candidate_probe_summary: dict[str, Any] = {}
    for layer_index, stat in sorted(candidate_probe_stats.items()):
        probe_count = max(1.0, stat["probe_count"])
        candidate_probe_summary[str(layer_index)] = {
            "probe_count": int(stat["probe_count"]),
            "avg_max_prob": stat["max_prob_sum"] / probe_count,
            "meets_importance_rate": stat["meets_importance_count"] / probe_count,
            "meets_confidence_rate": stat["meets_confidence_count"] / probe_count,
            "exit_rate": stat["exit_count"] / probe_count,
        }

    return {
        "token_count": len(token_traces),
        "avg_exit_layer": avg_exit_layer,
        "avg_exit_ratio": avg_exit_layer / max(1.0, float(total_layers - 1)),
        "exit_triggered_tokens": exit_triggered_tokens,
        "forced_cap_tokens": forced_cap_tokens,
        "string_guard_blocked_tokens": string_guard_blocked_tokens,
        "draft_only_candidate_tokens": draft_only_candidate_tokens,
        "draft_verified_matches": draft_verified_matches,
        "draft_verified_mismatches": draft_verified_mismatches,
        "max_prob_mean": _mean(max_probs),
        "token_exit_layer_histogram": {str(layer): int(count) for layer, count in sorted(histogram.items())},
        "candidate_probe_summary": candidate_probe_summary,
    }


def _resolve_eos_token_ids(eos_token_id: int | list[int] | tuple[int, ...] | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(value) for value in eos_token_id}


def _sample_next_token(probs: torch.Tensor, *, do_sample: bool) -> int:
    if probs.ndim != 1:
        raise ValueError("probs 必须是一维向量。")
    if do_sample:
        sampled = torch.multinomial(probs, num_samples=1)
        return int(sampled.item())
    return int(torch.argmax(probs).item())


def _resolve_causal_lm_modules(model: Any) -> tuple[Any, Any, Any]:
    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    config = getattr(model, "config", None)
    if backbone is None or lm_head is None or config is None:
        raise ValueError("当前模型不具备早退所需的 CausalLM 结构。")

    required_backbone_attrs = ("embed_tokens", "layers", "norm", "rotary_emb")
    missing = [name for name in required_backbone_attrs if not hasattr(backbone, name)]
    if missing:
        raise ValueError(f"当前模型 backbone 缺少早退所需属性: {missing}")

    model_type = str(getattr(config, "model_type", "")).strip().lower()
    if model_type != "qwen2":
        raise ValueError(f"当前最小实现仅支持 qwen2/qwen2.5 系列模型，实际为: {model_type or 'unknown'}")

    return backbone, lm_head, config


@dataclass
class EarlyExitGenerationConfig:
    max_new_tokens: int
    temperature: float
    eos_token_id: int | list[int] | tuple[int, ...] | None
    pad_token_id: int | None = None
    warmup_tokens: int = 16
    min_exit_streak: int = 4
    protect_open_string: bool = False
    draft_only_exit_layers: tuple[int, ...] = ()

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


class Qwen2EarlyExitGenerator:
    """
    面向 Qwen2/Qwen2.5 的最小动态早退生成器。

    说明：
    1. 为保持 KV cache 一致性，当前实现采用“有效深度单调不升”的策略；
    2. 一旦某个 token 在较浅层退出，后续 token 的最大可用深度不会再升高；
    3. 这是一个工程化 MVP，优先保证真实停层与现有 HF/PEFT 兼容。
    """

    def __init__(
        self,
        *,
        model: Any,
        policy: EarlyExitPolicy,
        generation_config: EarlyExitGenerationConfig,
        torch_module: Any,
        tokenizer: Any | None = None,
    ) -> None:
        self.model = model
        self.policy = policy
        self.generation_config = generation_config
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.backbone, self.lm_head, self.config = _resolve_causal_lm_modules(model)
        self.total_layers = int(len(self.backbone.layers))
        if self.total_layers <= 0:
            raise ValueError("模型层数必须大于 0。")

        if self.policy.total_layers != self.total_layers:
            raise ValueError(
                f"退出策略层数与模型不一致：policy={self.policy.total_layers}, model={self.total_layers}"
            )

        try:
            from transformers.cache_utils import DynamicCache
            from transformers.models.qwen2.modeling_qwen2 import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 transformers 依赖，无法运行早退生成。") from exc

        self.DynamicCache = DynamicCache
        self.create_causal_mask = create_causal_mask
        self.create_sliding_window_causal_mask = create_sliding_window_causal_mask
        self.draft_only_exit_layers = {int(layer) for layer in self.generation_config.draft_only_exit_layers}

    def _decode_token_text(self, token_id: int) -> str:
        if self.tokenizer is None:
            return ""
        try:
            return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=False))
        except TypeError:
            return str(self.tokenizer.decode([int(token_id)]))

    def _build_attention_mapping(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
    ) -> dict[str, torch.Tensor]:
        if isinstance(attention_mask, dict):
            return attention_mask

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        mask_mapping = {
            "full_attention": self.create_causal_mask(**mask_kwargs),
        }
        if bool(getattr(self.backbone, "has_sliding_layers", False)):
            mask_mapping["sliding_attention"] = self.create_sliding_window_causal_mask(**mask_kwargs)
        return mask_mapping

    def _prepare_forward_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = self.DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        position_ids = cache_position.unsqueeze(0)
        attention_mapping = self._build_attention_mapping(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return past_key_values, inputs_embeds, cache_position, position_ids, attention_mapping

    def _project_to_probs(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, float]:
        normed = self.backbone.norm(hidden_states)
        logits = self.lm_head(normed[:, -1:, :])[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        max_prob = float(torch.max(probs).item()) if probs.numel() > 0 else 0.0
        return probs, max_prob

    def _forward_until_exit(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        current_layer_cap: int,
        token_index: int,
    ) -> tuple[int, dict[str, Any], Any]:
        (
            past_key_values,
            hidden_states,
            cache_position,
            position_ids,
            attention_mapping,
        ) = self._prepare_forward_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        position_embeddings = self.backbone.rotary_emb(hidden_states, position_ids)
        effective_exit_layers = set(self.policy.effective_exit_layers(layer_cap=current_layer_cap))

        selected_probs: torch.Tensor | None = None
        selected_trace: dict[str, Any] | None = None
        candidate_exit_layer: int | None = None
        probe_details: list[dict[str, Any]] = []
        draft_token_id_by_layer: dict[int, int] = {}

        for layer_index, decoder_layer in enumerate(self.backbone.layers):
            if layer_index > current_layer_cap:
                break

            attention_type = str(getattr(decoder_layer, "attention_type", "full_attention"))
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mapping[attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if layer_index not in effective_exit_layers:
                continue

            probs, max_prob = self._project_to_probs(hidden_states)
            draft_token_id_by_layer[layer_index] = int(torch.argmax(probs[0]).item())
            probe = self.policy.probe(layer_index=layer_index, max_prob=max_prob)
            probe_details.append(probe.to_dict())
            if probe.should_exit and candidate_exit_layer is None:
                candidate_exit_layer = layer_index

        selected_probs, max_prob = self._project_to_probs(hidden_states)
        probe = self.policy.probe(layer_index=current_layer_cap, max_prob=max_prob)
        selected_trace = {
            **probe.to_dict(),
            "token_index": token_index,
            "exit_layer": current_layer_cap,
            "layer_cap_before_token": current_layer_cap,
            "forced_by_cache_cap": current_layer_cap < (self.total_layers - 1),
            "exit_triggered": candidate_exit_layer is not None,
            "candidate_exit_layer": candidate_exit_layer,
            "probe_details": probe_details,
            "draft_token_id_by_layer": {str(layer): token_id for layer, token_id in sorted(draft_token_id_by_layer.items())},
        }

        temperature = max(1e-5, float(self.generation_config.temperature))
        if self.generation_config.do_sample:
            sampling_probs = torch.softmax(torch.log(selected_probs + 1e-12) / temperature, dim=-1)
        else:
            sampling_probs = selected_probs
        token_id = _sample_next_token(sampling_probs[0], do_sample=self.generation_config.do_sample)
        selected_trace["token_id"] = token_id
        return token_id, selected_trace, past_key_values

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids 与 attention_mask 必须是二维张量。")
        if int(input_ids.shape[0]) != 1:
            raise ValueError("当前最小早退实现仅支持 batch_size=1。")

        generated_ids = input_ids
        running_attention_mask = attention_mask
        past_key_values: Any = None
        token_traces: list[dict[str, Any]] = []
        current_layer_cap = self.total_layers - 1
        eos_token_ids = _resolve_eos_token_ids(self.generation_config.eos_token_id)
        last_input_ids = input_ids
        exit_streak = 0
        streak_layer: int | None = None
        in_open_string = False
        pending_escape = False

        for token_index in range(int(self.generation_config.max_new_tokens)):
            in_open_string_before_token = in_open_string
            pending_escape_before_token = pending_escape
            next_token_id, token_trace, past_key_values = self._forward_until_exit(
                input_ids=last_input_ids,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                current_layer_cap=current_layer_cap,
                token_index=token_index,
            )
            token_traces.append(token_trace)
            candidate_exit_layer = token_trace.get("candidate_exit_layer")
            string_guard_blocked_exit = False
            draft_only_candidate = False
            draft_verified_match = False
            draft_verified_mismatch = False
            chosen_candidate_layer: int | None = None
            if (
                isinstance(candidate_exit_layer, int)
                and candidate_exit_layer < current_layer_cap
                and token_index >= int(self.generation_config.warmup_tokens)
            ):
                if bool(self.generation_config.protect_open_string) and in_open_string:
                    string_guard_blocked_exit = True
                    streak_layer = None
                    exit_streak = 0
                elif candidate_exit_layer in self.draft_only_exit_layers:
                    draft_only_candidate = True
                    chosen_candidate_layer = int(candidate_exit_layer)
                    streak_layer = None
                    exit_streak = 0
                else:
                    chosen_candidate_layer = int(candidate_exit_layer)
                    if streak_layer == candidate_exit_layer:
                        exit_streak += 1
                    else:
                        streak_layer = candidate_exit_layer
                        exit_streak = 1
            else:
                streak_layer = None
                exit_streak = 0

            cap_updated = False
            if (
                streak_layer is not None
                and exit_streak >= int(self.generation_config.min_exit_streak)
                and streak_layer < current_layer_cap
            ):
                current_layer_cap = int(streak_layer)
                cap_updated = True
                streak_layer = None
                exit_streak = 0

            selected_token_id = int(token_trace.get("token_id", next_token_id))
            if draft_only_candidate and chosen_candidate_layer is not None:
                draft_token_id_by_layer = token_trace.get("draft_token_id_by_layer", {})
                raw_draft_token_id = (
                    draft_token_id_by_layer.get(str(chosen_candidate_layer))
                    if isinstance(draft_token_id_by_layer, dict)
                    else None
                )
                if raw_draft_token_id is not None:
                    draft_verified_match = int(raw_draft_token_id) == selected_token_id
                    draft_verified_mismatch = not draft_verified_match
                    token_trace["draft_token_id"] = int(raw_draft_token_id)
                    token_trace["draft_token_text"] = self._decode_token_text(int(raw_draft_token_id))

            token_text = self._decode_token_text(selected_token_id)
            in_open_string, pending_escape = _update_string_state(
                token_text,
                in_open_string=in_open_string,
                pending_escape=pending_escape,
            )

            token_trace["cap_update_applied_for_next_token"] = cap_updated
            token_trace["next_layer_cap"] = current_layer_cap
            token_trace["warmup_tokens"] = int(self.generation_config.warmup_tokens)
            token_trace["min_exit_streak"] = int(self.generation_config.min_exit_streak)
            token_trace["string_guard_enabled"] = bool(self.generation_config.protect_open_string)
            token_trace["string_guard_blocked_exit"] = string_guard_blocked_exit
            token_trace["in_open_string_before_token"] = in_open_string_before_token
            token_trace["in_open_string_after_token"] = in_open_string
            token_trace["pending_escape_before_token"] = pending_escape_before_token
            token_trace["pending_escape_after_token"] = pending_escape
            token_trace["token_text"] = token_text
            token_trace["draft_only_candidate"] = draft_only_candidate
            token_trace["draft_only_exit_layers"] = sorted(self.draft_only_exit_layers)
            token_trace["draft_verified_match"] = draft_verified_match
            token_trace["draft_verified_mismatch"] = draft_verified_mismatch

            next_token_tensor = generated_ids.new_tensor([[next_token_id]])
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
            last_input_ids = next_token_tensor
            running_attention_mask = torch.cat(
                [running_attention_mask, running_attention_mask.new_ones((1, 1))],
                dim=1,
            )

            if next_token_id in eos_token_ids:
                break

        summary = summarize_token_exit_traces(token_traces, total_layers=self.total_layers)
        trace = {
            "early_exit_enabled": True,
            "algorithm": "IIDEE_MVP",
            "total_layers": self.total_layers,
            "exit_layers": list(self.policy.exit_layers),
            "tau_importance": float(self.policy.tau_importance),
            "tau_confidence": float(self.policy.tau_confidence),
            "monotonic_layer_cap_enabled": True,
            "warmup_tokens": int(self.generation_config.warmup_tokens),
            "min_exit_streak": int(self.generation_config.min_exit_streak),
            "protect_open_string": bool(self.generation_config.protect_open_string),
            "draft_only_exit_layers": sorted(self.draft_only_exit_layers),
            "tokens_generated": len(token_traces),
            "final_layer_cap": current_layer_cap,
            "token_traces": token_traces,
            "summary": summary,
        }
        return generated_ids, trace
