#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import inspect
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

QuantizationMode = Literal["awq", "4bit", "8bit"]
BackendMode = Literal["auto", "vllm", "transformers"]


@dataclass
class _TransformersPrefixCacheEntry:
    prefix_text: str
    prefix_token_ids: Any
    prefix_token_count: int
    past_key_values: Any


class LocalLLMEngine:
    """Local text-generation engine with switchable inference backend."""

    def __init__(
        self,
        model_path: str,
        backend: BackendMode = "auto",
        quantization: QuantizationMode | None = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        default_temperature: float = 0.0,
        default_top_p: float = 1.0,
        default_max_new_tokens: int = 512,
        enable_prefix_caching: bool = False,
        prefix_cache_max_entries: int = 4,
    ) -> None:
        self.model_path = str(model_path)
        self.backend_mode = backend
        self.quantization = quantization
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.trust_remote_code = bool(trust_remote_code)
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.prefix_cache_max_entries = int(prefix_cache_max_entries)

        self.backend: Literal["vllm", "transformers"] | None = None
        self._vllm_engine: Any | None = None
        self._hf_model: Any | None = None
        self._tokenizer: Any | None = None
        self._transformers_prefix_cache: OrderedDict[str, _TransformersPrefixCacheEntry] = OrderedDict()

        self._validate_init_args()
        self._initialize_backend()

    def _prefix_profile_enabled(self) -> bool:
        value = os.getenv("LLM_GENESIS_PREFIX_PROFILE", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _emit_prefix_profile(self, **payload: Any) -> None:
        if not self._prefix_profile_enabled():
            return
        parts = ["[prefix-profile]"]
        for key, value in payload.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts), file=sys.stderr, flush=True)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Run generation and return raw text."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("`prompt` must be a non-empty string.")

        resolved_temperature, resolved_top_p, resolved_max_new_tokens = self._resolve_generation_args(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        if self.backend == "vllm":
            raw_text = self._generate_with_vllm(
                prompt=prompt,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                max_new_tokens=resolved_max_new_tokens,
            )
        elif self.backend == "transformers":
            raw_text = self._generate_with_transformers(
                prompt=prompt,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                max_new_tokens=resolved_max_new_tokens,
            )
        else:
            raise RuntimeError("No backend is initialized. Cannot generate text.")
        return raw_text

    def generate_with_prefix(
        self,
        prefix_prompt: str,
        suffix_prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        cache_key: str | None = None,
    ) -> str:
        """Generate from a prompt split into cacheable prefix and dynamic suffix."""
        if not isinstance(prefix_prompt, str):
            raise ValueError("`prefix_prompt` must be a string.")
        if not isinstance(suffix_prompt, str) or not suffix_prompt.strip():
            raise ValueError("`suffix_prompt` must be a non-empty string.")

        resolved_temperature, resolved_top_p, resolved_max_new_tokens = self._resolve_generation_args(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        full_prompt = f"{prefix_prompt}{suffix_prompt}"
        if not prefix_prompt or not self.enable_prefix_caching:
            return self.generate(
                prompt=full_prompt,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                max_new_tokens=resolved_max_new_tokens,
            )

        if self.backend == "transformers":
            return self._generate_with_transformers_prefix_cache(
                prefix_prompt=prefix_prompt,
                suffix_prompt=suffix_prompt,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                max_new_tokens=resolved_max_new_tokens,
                cache_key=cache_key,
            )

        if self.backend == "vllm":
            # vLLM 在引擎初始化时打开 enable_prefix_caching 后，会自动识别公共前缀。
            return self._generate_with_vllm(
                prompt=full_prompt,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                max_new_tokens=resolved_max_new_tokens,
            )

        raise RuntimeError("No backend is initialized. Cannot generate text.")

    def warm_prefix(self, prefix_prompt: str, *, cache_key: str | None = None) -> bool:
        """Pre-compute and retain prefix KV cache for the transformers backend."""
        if not self.enable_prefix_caching or self.backend != "transformers":
            return False
        if not isinstance(prefix_prompt, str) or not prefix_prompt:
            return False

        started_at = time.perf_counter()
        self._get_or_create_transformers_prefix_cache(
            prefix_prompt=prefix_prompt,
            cache_key=cache_key,
        )
        self._emit_prefix_profile(
            stage="warm_prefix",
            cache_key=cache_key or "-",
            prefix_chars=len(prefix_prompt),
            elapsed_ms=f"{(time.perf_counter() - started_at) * 1000:.2f}",
        )
        return True

    def clear_prefix_cache(self) -> None:
        self._transformers_prefix_cache.clear()

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("`prompts` must be a non-empty list of strings.")
        resolved_prompts = [str(p) for p in prompts]
        if any(not p.strip() for p in resolved_prompts):
            raise ValueError("`prompts` contains empty prompt.")
        return [
            self.generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            for prompt in resolved_prompts
        ]

    def _resolve_generation_args(
        self,
        *,
        temperature: float | None,
        top_p: float | None,
        max_new_tokens: int | None,
    ) -> tuple[float, float, int]:
        resolved_temperature = self.default_temperature if temperature is None else float(temperature)
        resolved_top_p = self.default_top_p if top_p is None else float(top_p)
        resolved_max_new_tokens = (
            self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        )

        if resolved_temperature < 0.0:
            raise ValueError("`temperature` must be >= 0.0.")
        if not (0.0 < resolved_top_p <= 1.0):
            raise ValueError("`top_p` must be in (0.0, 1.0].")
        if resolved_max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be > 0.")
        return resolved_temperature, resolved_top_p, resolved_max_new_tokens

    def _validate_init_args(self) -> None:
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        supported_backend = {"auto", "vllm", "transformers"}
        if self.backend_mode not in supported_backend:
            raise ValueError(
                f"Unsupported backend={self.backend_mode!r}. "
                "Supported values: 'auto', 'vllm', 'transformers'."
            )

        supported_quant = {None, "awq", "4bit", "8bit"}
        if self.quantization not in supported_quant:
            raise ValueError(
                f"Unsupported quantization={self.quantization!r}. "
                "Supported values: None, 'awq', '4bit', '8bit'."
            )

        if self.max_model_len <= 0:
            raise ValueError("`max_model_len` must be > 0.")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError("`gpu_memory_utilization` must be in (0.0, 1.0].")
        if self.default_temperature < 0.0:
            raise ValueError("`default_temperature` must be >= 0.0.")
        if not (0.0 < self.default_top_p <= 1.0):
            raise ValueError("`default_top_p` must be in (0.0, 1.0].")
        if self.default_max_new_tokens <= 0:
            raise ValueError("`default_max_new_tokens` must be > 0.")
        if self.prefix_cache_max_entries <= 0:
            raise ValueError("`prefix_cache_max_entries` must be > 0.")

    def _initialize_backend(self) -> None:
        init_errors: list[str] = []

        backend_order: list[Literal["vllm", "transformers"]]
        if self.backend_mode == "vllm":
            backend_order = ["vllm"]
        elif self.backend_mode == "transformers":
            backend_order = ["transformers"]
        else:
            backend_order = ["vllm", "transformers"]

        for backend in backend_order:
            try:
                if backend == "vllm":
                    self._init_vllm()
                else:
                    self._init_transformers()
                self.backend = backend
                return
            except Exception as exc:
                init_errors.append(f"{backend} init failed: {type(exc).__name__}: {exc}")

        err_msg = (
            f"Failed to initialize local inference backend (mode={self.backend_mode!r}).\n"
            + "\n".join(init_errors)
        )
        raise RuntimeError(err_msg)

    def _init_vllm(self) -> None:
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": "auto",
        }

        if self.quantization == "awq":
            kwargs["quantization"] = "awq"
        elif self.quantization in {"4bit", "8bit"}:
            # Let vLLM handle bitsandbytes when available; otherwise fallback
            # to transformers in `_initialize_backend`.
            kwargs["quantization"] = "bitsandbytes"
            kwargs["load_format"] = "bitsandbytes"

        if self.enable_prefix_caching:
            llm_init_sig = inspect.signature(LLM.__init__)
            if "enable_prefix_caching" in llm_init_sig.parameters:
                kwargs["enable_prefix_caching"] = True

        self._vllm_engine = LLM(**kwargs)

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        if self.quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise ImportError(
                    "bitsandbytes quantization requested, but BitsAndBytesConfig "
                    "is unavailable. Please install `transformers` + `bitsandbytes`."
                ) from exc

            if self.quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._hf_model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._hf_model.eval()

    def _generate_with_vllm(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        if self._vllm_engine is None:
            raise RuntimeError("vLLM backend is not initialized.")

        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self._vllm_engine.generate([prompt], sampling_params=sampling, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty generation output.")

        text = outputs[0].outputs[0].text.strip()
        if not text:
            raise RuntimeError("vLLM generated empty text.")
        return text

    def _generate_with_transformers(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        if self._hf_model is None or self._tokenizer is None:
            raise RuntimeError("transformers backend is not initialized.")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        return self._generate_transformers_from_inputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    def _generate_with_transformers_prefix_cache(
        self,
        *,
        prefix_prompt: str,
        suffix_prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        cache_key: str | None,
    ) -> str:
        if self._hf_model is None or self._tokenizer is None:
            raise RuntimeError("transformers backend is not initialized.")

        full_prompt = f"{prefix_prompt}{suffix_prompt}"
        tokenize_started_at = time.perf_counter()
        inputs, prefix_token_count = self._tokenize_with_prefix_boundary(
            prompt=full_prompt,
            prefix_prompt=prefix_prompt,
        )
        tokenize_elapsed_ms = (time.perf_counter() - tokenize_started_at) * 1000

        if prefix_token_count is None or prefix_token_count <= 0:
            return self._generate_transformers_from_inputs(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )

        if inputs["input_ids"].shape[-1] <= prefix_token_count:
            return self._generate_transformers_from_inputs(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )

        cache_started_at = time.perf_counter()
        entry = self._get_or_create_transformers_prefix_cache(
            prefix_prompt=prefix_prompt,
            cache_key=cache_key,
            prefix_input_ids=inputs["input_ids"][:, :prefix_token_count],
            prefix_attention_mask=inputs["attention_mask"][:, :prefix_token_count],
        )
        cache_elapsed_ms = (time.perf_counter() - cache_started_at) * 1000

        import torch

        suffix_input_ids = inputs["input_ids"][:, prefix_token_count:]
        full_attention_mask = inputs["attention_mask"]
        target_device = self._get_hf_input_device()

        suffix_input_ids = suffix_input_ids.to(target_device)
        full_attention_mask = full_attention_mask.to(target_device)
        cache_position = torch.arange(
            prefix_token_count,
            inputs["input_ids"].shape[-1],
            device=target_device,
        )

        generate_kwargs = self._build_transformers_generate_kwargs(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        deepcopy_started_at = time.perf_counter()
        past_key_values = copy.deepcopy(entry.past_key_values)
        deepcopy_elapsed_ms = (time.perf_counter() - deepcopy_started_at) * 1000

        generate_started_at = time.perf_counter()
        with torch.no_grad():
            generated_ids = self._hf_model.generate(
                input_ids=suffix_input_ids,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **generate_kwargs,
            )
        generate_elapsed_ms = (time.perf_counter() - generate_started_at) * 1000

        suffix_len = suffix_input_ids.shape[-1]
        decode_started_at = time.perf_counter()
        new_token_ids = generated_ids[0][suffix_len:]
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        decode_elapsed_ms = (time.perf_counter() - decode_started_at) * 1000
        self._emit_prefix_profile(
            stage="generate_with_prefix",
            cache_key=cache_key or "-",
            cache_type=type(entry.past_key_values).__name__,
            prefix_chars=len(prefix_prompt),
            prefix_tokens=prefix_token_count,
            suffix_tokens=int(suffix_input_ids.shape[-1]),
            tokenize_ms=f"{tokenize_elapsed_ms:.2f}",
            cache_lookup_ms=f"{cache_elapsed_ms:.2f}",
            deepcopy_ms=f"{deepcopy_elapsed_ms:.2f}",
            generate_ms=f"{generate_elapsed_ms:.2f}",
            decode_ms=f"{decode_elapsed_ms:.2f}",
        )
        if not text:
            raise RuntimeError("transformers generated empty text.")
        return text

    def _generate_transformers_from_inputs(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        if self._hf_model is None or self._tokenizer is None:
            raise RuntimeError("transformers backend is not initialized.")

        import torch

        target_device = self._get_hf_input_device()
        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)

        with torch.no_grad():
            generated_ids = self._hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self._build_transformers_generate_kwargs(
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                ),
            )

        prompt_len = input_ids.shape[-1]
        new_token_ids = generated_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("transformers generated empty text.")
        return text

    def _build_transformers_generate_kwargs(
        self,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer is not initialized.")

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        return generate_kwargs

    def _tokenize_with_prefix_boundary(
        self,
        *,
        prompt: str,
        prefix_prompt: str,
    ) -> tuple[dict[str, Any], int | None]:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer is not initialized.")

        tokenized: dict[str, Any] = self._tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized.pop("offset_mapping", None)
        prefix_token_count = self._resolve_prefix_token_count(
            prompt=prompt,
            prefix_prompt=prefix_prompt,
            full_input_ids=tokenized["input_ids"],
            offset_mapping=offset_mapping,
        )
        return tokenized, prefix_token_count

    def _resolve_prefix_token_count(
        self,
        *,
        prompt: str,
        prefix_prompt: str,
        full_input_ids: Any,
        offset_mapping: Any,
    ) -> int | None:
        if not prompt.startswith(prefix_prompt):
            return None

        prefix_boundary = len(prefix_prompt)
        if offset_mapping is not None:
            boundary = self._prefix_token_boundary_from_offsets(
                offset_mapping=offset_mapping,
                prefix_boundary=prefix_boundary,
            )
            if boundary is not None:
                return boundary

        if self._tokenizer is None:
            return None

        prefix_inputs = self._tokenizer(prefix_prompt, return_tensors="pt")
        prefix_ids = prefix_inputs["input_ids"]
        prefix_token_count = prefix_ids.shape[-1]
        if full_input_ids.shape[-1] < prefix_token_count:
            return None
        if full_input_ids[:, :prefix_token_count].equal(prefix_ids):
            return prefix_token_count
        return None

    def _prefix_token_boundary_from_offsets(
        self,
        *,
        offset_mapping: Any,
        prefix_boundary: int,
    ) -> int | None:
        row_offsets = offset_mapping[0].tolist()
        prefix_token_count = 0
        for start, end in row_offsets:
            if start == end == 0 and prefix_token_count == 0:
                prefix_token_count += 1
                continue
            if end <= prefix_boundary:
                prefix_token_count += 1
                continue
            if start < prefix_boundary < end:
                return None
            break
        return prefix_token_count

    def _get_or_create_transformers_prefix_cache(
        self,
        *,
        prefix_prompt: str,
        cache_key: str | None,
        prefix_input_ids: Any | None = None,
        prefix_attention_mask: Any | None = None,
    ) -> _TransformersPrefixCacheEntry:
        if self._hf_model is None or self._tokenizer is None:
            raise RuntimeError("transformers backend is not initialized.")

        key = self._prefix_cache_lookup_key(prefix_prompt=prefix_prompt, cache_key=cache_key)
        cached = self._transformers_prefix_cache.get(key)
        if cached is not None and cached.prefix_text == prefix_prompt:
            if prefix_input_ids is None or cached.prefix_token_ids.equal(prefix_input_ids.detach().cpu()):
                self._transformers_prefix_cache.move_to_end(key)
                return cached

        if prefix_input_ids is None or prefix_attention_mask is None:
            tokenized = self._tokenizer(prefix_prompt, return_tensors="pt")
            prefix_input_ids = tokenized["input_ids"]
            prefix_attention_mask = tokenized["attention_mask"]

        target_device = self._get_hf_input_device()
        model_inputs = {
            "input_ids": prefix_input_ids.to(target_device),
            "attention_mask": prefix_attention_mask.to(target_device),
        }

        import torch

        with torch.no_grad():
            outputs = self._hf_model(**model_inputs, use_cache=True)

        entry = _TransformersPrefixCacheEntry(
            prefix_text=prefix_prompt,
            prefix_token_ids=prefix_input_ids.detach().cpu(),
            prefix_token_count=int(prefix_input_ids.shape[-1]),
            past_key_values=outputs.past_key_values,
        )
        self._transformers_prefix_cache[key] = entry
        self._transformers_prefix_cache.move_to_end(key)
        while len(self._transformers_prefix_cache) > self.prefix_cache_max_entries:
            self._transformers_prefix_cache.popitem(last=False)
        return entry

    def _prefix_cache_lookup_key(self, *, prefix_prompt: str, cache_key: str | None) -> str:
        if cache_key is not None and cache_key.strip():
            return f"user::{cache_key.strip()}"
        digest = hashlib.sha1(prefix_prompt.encode("utf-8")).hexdigest()
        return f"sha1::{digest}"

    def _get_hf_input_device(self) -> Any:
        if self._hf_model is None:
            raise RuntimeError("transformers model is not initialized.")

        if hasattr(self._hf_model, "device") and self._hf_model.device is not None:
            return self._hf_model.device
        return next(self._hf_model.parameters()).device
