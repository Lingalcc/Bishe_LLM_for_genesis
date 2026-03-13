#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_model_path(model_path: str | Path) -> str:
    """将模型路径规范化为绝对路径字符串。"""
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


class HFInferenceEngine:
    """
    基于 HuggingFace Transformers 的本地推理引擎。

    支持模式：
    1) FP16 baseline
    2) bitsandbytes 4bit/8bit 量化
    """

    def __init__(
        self,
        *,
        model_path: str,
        quantization: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.quantization = quantization
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.trust_remote_code = bool(trust_remote_code)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 transformers/torch 依赖，无法运行 transformers 推理。") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto",
        }

        if self.quantization in {None, "", "none"}:
            model_kwargs["torch_dtype"] = torch.float16
        elif self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ModuleNotFoundError as exc:
                raise RuntimeError("请求 4bit 量化，但当前环境缺少 bitsandbytes 支持。") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["torch_dtype"] = torch.float16
        elif self.quantization == "8bit":
            try:
                from transformers import BitsAndBytesConfig
            except ModuleNotFoundError as exc:
                raise RuntimeError("请求 8bit 量化，但当前环境缺少 bitsandbytes 支持。") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["torch_dtype"] = torch.float16
        else:
            raise ValueError(
                f"不支持的 quantization={self.quantization!r}。"
                "transformers backend 仅支持: None/'4bit'/'8bit'"
            )

        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._model.eval()

    def _input_device(self) -> Any:
        if hasattr(self._model, "device") and self._model.device is not None:
            return self._model.device
        return next(self._model.parameters()).device

    def _generate_from_tensors(self, input_ids: Any, attention_mask: Any) -> list[str]:
        """Run batched generation on tokenized tensors."""
        target_device = self._input_device()
        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        with self._torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs,
            )

        prompt_lens = attention_mask.sum(dim=1).tolist()
        texts: list[str] = []
        for row_idx, prompt_len in enumerate(prompt_lens):
            new_ids = output_ids[row_idx][int(prompt_len):]
            texts.append(self._tokenizer.decode(new_ids, skip_special_tokens=True).strip())
        return texts

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串。")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._generate_from_tensors(inputs["input_ids"], inputs["attention_mask"])
        return outputs[0]

    @staticmethod
    def _validate_prompts(prompts: list[str]) -> list[str]:
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts 必须是非空字符串列表。")
        normalized = [str(p) for p in prompts]
        if any(not p.strip() for p in normalized):
            raise ValueError("prompts 中包含空字符串。")
        return normalized

    def generate_batch(self, prompts: list[str]) -> list[str]:
        prompts = self._validate_prompts(prompts)
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
        return self._generate_from_tensors(inputs["input_ids"], inputs["attention_mask"])

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        """Generate using chat template for Instruct models."""
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        outputs = self._generate_from_tensors(inputs["input_ids"], inputs["attention_mask"])
        return outputs[0]

    def generate_chat_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        prompts = [
            self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        return self.generate_batch(prompts)


class VLLMInferenceEngine:
    """基于 vLLM 的推理引擎。"""

    def __init__(
        self,
        *,
        model_path: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        quantization: str | None = None,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.trust_remote_code = bool(trust_remote_code)
        self.quantization = quantization

        try:
            from vllm import LLM, SamplingParams
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 vllm 依赖，无法运行 vLLM 推理。") from exc

        llm_kwargs: dict[str, Any] = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "dtype": "auto",
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.quantization:
            llm_kwargs["quantization"] = self.quantization
            if self.quantization == "bitsandbytes":
                llm_kwargs["load_format"] = "bitsandbytes"

        self._SamplingParams = SamplingParams
        self._engine = LLM(**llm_kwargs)

    def _sampling_params(self) -> Any:
        return self._SamplingParams(
            temperature=self.temperature,
            top_p=0.95 if self.temperature > 0 else 1.0,
            max_tokens=self.max_new_tokens,
        )

    @staticmethod
    def _validate_prompts(prompts: list[str]) -> list[str]:
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts 必须是非空字符串列表。")
        normalized = [str(p) for p in prompts]
        if any(not p.strip() for p in normalized):
            raise ValueError("prompts 中包含空字符串。")
        return normalized

    def _generate_many(self, prompts: list[str]) -> list[str]:
        prompts = self._validate_prompts(prompts)
        outputs = self._engine.generate(prompts, sampling_params=self._sampling_params(), use_tqdm=False)
        texts: list[str] = []
        for item in outputs:
            if not item.outputs:
                raise RuntimeError("vLLM 输出为空。")
            texts.append(item.outputs[0].text.strip())
        return texts

    def generate(self, prompt: str) -> str:
        return self._generate_many([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return self._generate_many(prompts)

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        """Generate using chat template for Instruct models."""
        tokenizer = self._engine.get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self._generate_many([text])[0]

    def generate_chat_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        tokenizer = self._engine.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        return self._generate_many(prompts)


def build_inference_engine(config: dict[str, Any]) -> Any:
    """
    根据配置动态创建推理引擎（工厂模式）。

    参数示例：
    {
      "backend": "transformers",
      "model_path": "model/my_model",
      "quantization": null,
      "max_new_tokens": 256,
      "temperature": 0.0
    }
    """
    if not isinstance(config, dict):
        raise TypeError("build_inference_engine 的 config 必须是 dict。")

    backend = str(config.get("backend", "")).strip().lower()
    model_path = config.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("配置项 `model_path` 必填。")

    if backend == "transformers":
        return HFInferenceEngine(
            model_path=model_path,
            quantization=config.get("quantization"),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            temperature=float(config.get("temperature", 0.0)),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
        )

    if backend == "vllm":
        return VLLMInferenceEngine(
            model_path=model_path,
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            temperature=float(config.get("temperature", 0.0)),
            max_model_len=int(config.get("max_model_len", 4096)),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            quantization=config.get("quantization"),
        )

    raise ValueError(f"不支持的 backend={backend!r}。可选值: 'transformers', 'vllm'")
