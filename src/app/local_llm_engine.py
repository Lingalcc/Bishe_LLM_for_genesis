#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

QuantizationMode = Literal["awq", "4bit", "8bit"]
BackendMode = Literal["auto", "vllm", "transformers"]


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

        self.backend: Literal["vllm", "transformers"] | None = None
        self._vllm_engine: Any | None = None
        self._hf_model: Any | None = None
        self._tokenizer: Any | None = None

        self._validate_init_args()
        self._initialize_backend()

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

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")

        target_device = self._get_hf_input_device()
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated_ids = self._hf_model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_token_ids = generated_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("transformers generated empty text.")
        return text

    def _get_hf_input_device(self) -> Any:
        if self._hf_model is None:
            raise RuntimeError("transformers model is not initialized.")

        if hasattr(self._hf_model, "device") and self._hf_model.device is not None:
            return self._hf_model.device
        return next(self._hf_model.parameters()).device
