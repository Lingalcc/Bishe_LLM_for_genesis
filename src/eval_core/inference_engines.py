#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.eval_core.performance_monitor import estimate_tokens_from_text


def _resolve_model_path(model_path: str | Path) -> str:
    """将模型路径规范化为绝对路径字符串。"""
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_model_and_adapter_paths(model_path: str | Path) -> tuple[str, str | None, dict[str, Any] | None]:
    resolved_model_path = _resolve_model_path(model_path)
    adapter_config_path = Path(resolved_model_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return resolved_model_path, None, None

    adapter_config = _load_json_dict(adapter_config_path)
    if not adapter_config:
        raise ValueError(f"检测到 PEFT adapter 目录，但无法解析配置文件: {adapter_config_path}")

    base_model_name_or_path = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model_name_or_path, str) or not base_model_name_or_path.strip():
        raise ValueError(f"adapter_config.json 缺少 base_model_name_or_path: {adapter_config_path}")

    return _resolve_model_path(base_model_name_or_path), resolved_model_path, adapter_config


def _normalize_quantization(quantization: str | None) -> str | None:
    if quantization is None:
        return None
    text = str(quantization).strip().lower()
    if text in {"", "none", "null"}:
        return None
    aliases = {
        "bitsandbytes_4bit": "4bit",
        "bnb_4bit": "4bit",
        "bitsandbytes_8bit": "8bit",
        "bnb_8bit": "8bit",
        "bitsandbytes": "bitsandbytes",
        "gguf_q4_k_m": "gguf_q4_k_m",
        "exl2_4bpw": "exl2_4bpw",
    }
    return aliases.get(text, text)


def _resolve_tokenizer_path(
    model_path: str | Path,
    tokenizer_path: str | Path | None,
) -> str | None:
    candidate = tokenizer_path if tokenizer_path else model_path
    try:
        resolved = Path(_resolve_model_path(candidate))
    except Exception:
        return None
    return str(resolved) if resolved.exists() else None


def _fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for item in messages:
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _load_chat_tokenizer(
    *,
    model_path: str,
    tokenizer_path: str | None,
    trust_remote_code: bool,
) -> Any | None:
    resolved = _resolve_tokenizer_path(model_path, tokenizer_path)
    if not resolved:
        return None
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception:
        return None
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _render_chat_prompt(tokenizer: Any | None, messages: list[dict[str, str]]) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _fallback_chat_prompt(messages)


def _count_tokens_with_tokenizer(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    if not text:
        return 0

    try:
        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if isinstance(token_ids, list):
                return len(token_ids)
    except TypeError:
        pass
    except Exception:
        return None

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except TypeError:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except Exception:
            return None
    except Exception:
        return None

    input_ids = None
    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = getattr(encoded, "input_ids", None)

    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    return None


def _count_tokens_with_llama_cpp_engine(engine: Any, text: str) -> int | None:
    if not text or engine is None or not hasattr(engine, "tokenize"):
        return 0 if not text else None
    raw = text.encode("utf-8")
    for kwargs in ({"add_bos": False, "special": False}, {"add_bos": False}, {}):
        try:
            token_ids = engine.tokenize(raw, **kwargs)
            if isinstance(token_ids, list):
                return len(token_ids)
        except TypeError:
            continue
        except Exception:
            return None
    return None


def _ensure_cuda_available(backend_name: str) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{backend_name} GPU-only 推理需要可用的 torch + CUDA 环境。") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(f"{backend_name} GPU-only 推理要求 CUDA 可用，当前环境未检测到可用 GPU。")


class HFInferenceEngine:
    """
    基于 HuggingFace Transformers 的本地推理引擎。

    支持模式：
    1) FP16 baseline
    2) bitsandbytes 4bit/8bit 量化
    3) LoRA adapter 自动挂载
    """

    def __init__(
        self,
        *,
        model_path: str,
        quantization: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
        tokenizer_path: str | None = None,
        use_flash_attention: bool = False,
        require_gpu: bool = False,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.base_model_path, self.adapter_path, self.adapter_config = _resolve_model_and_adapter_paths(self.model_path)
        self.quantization = _normalize_quantization(quantization)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.trust_remote_code = bool(trust_remote_code)
        self.tokenizer_path = _resolve_tokenizer_path(self.model_path, tokenizer_path)
        self.use_flash_attention = bool(use_flash_attention)
        self.require_gpu = bool(require_gpu)

        if self.require_gpu:
            _ensure_cuda_available("transformers")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 transformers/torch 依赖，无法运行 transformers 推理。") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path or self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.require_gpu:
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["device_map"] = "auto"
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if self.quantization is None:
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
                " transformers backend 仅支持: None/'4bit'/'8bit'"
            )

        self._model = AutoModelForCausalLM.from_pretrained(self.base_model_path, **model_kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:
                raise RuntimeError("检测到 LoRA adapter，但当前环境缺少 peft 依赖。") from exc
            offload_dir = Path("/tmp/peft_offload") / Path(self.adapter_path).name
            offload_dir.mkdir(parents=True, exist_ok=True)
            self._model = PeftModel.from_pretrained(
                self._model,
                self.adapter_path,
                is_trainable=False,
                offload_dir=str(offload_dir),
            )
        self._model.eval()

    @property
    def token_count_method(self) -> str:
        return "tokenizer_exact"

    def _input_device(self) -> Any:
        if hasattr(self._model, "device") and self._model.device is not None:
            return self._model.device
        return next(self._model.parameters()).device

    def _generate_from_tensors(self, input_ids: Any, attention_mask: Any) -> list[str]:
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
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        prompt_lens = attention_mask.sum(dim=1).tolist()
        texts: list[str] = []
        for row_idx, prompt_len in enumerate(prompt_lens):
            new_ids = output_ids[row_idx][int(prompt_len) :]
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

    def render_prompt_for_metrics(self, prompt: str) -> str:
        return str(prompt)

    def render_batch_for_metrics(self, prompts: list[str]) -> list[str]:
        return [self.render_prompt_for_metrics(prompt) for prompt in prompts]

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        text = _render_chat_prompt(self._tokenizer, messages)
        return self.generate(text)

    def generate_chat_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        prompts = [_render_chat_prompt(self._tokenizer, msgs) for msgs in messages_batch]
        return self.generate_batch(prompts)

    def render_chat_for_metrics(self, messages: list[dict[str, str]]) -> str:
        return _render_chat_prompt(self._tokenizer, messages)

    def render_chat_batch_for_metrics(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        return [self.render_chat_for_metrics(messages) for messages in messages_batch]

    def count_tokens(self, text: str) -> int:
        count = _count_tokens_with_tokenizer(self._tokenizer, str(text))
        if count is not None:
            return count
        return estimate_tokens_from_text(str(text))


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
        tokenizer_path: str | None = None,
        require_gpu: bool = False,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.base_model_path, self.adapter_path, self.adapter_config = _resolve_model_and_adapter_paths(self.model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.trust_remote_code = bool(trust_remote_code)
        self.quantization = _normalize_quantization(quantization)
        self.tokenizer_path = _resolve_tokenizer_path(self.model_path, tokenizer_path)
        self.require_gpu = bool(require_gpu)

        # 当前项目里的 vLLM 接入默认视为 GPU-only，本地无 CUDA 时尽早给出清晰错误。
        _ensure_cuda_available("vllm")

        try:
            from vllm import LLM, SamplingParams
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 vllm 依赖，无法运行 vLLM 推理。") from exc
        try:
            from vllm.lora.request import LoRARequest
        except ModuleNotFoundError:
            LoRARequest = None

        llm_kwargs: dict[str, Any] = {
            "model": self.base_model_path,
            "trust_remote_code": self.trust_remote_code,
            "dtype": "auto",
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.tokenizer_path:
            llm_kwargs["tokenizer"] = self.tokenizer_path
        if self.quantization:
            if self.quantization == "4bit":
                llm_kwargs["quantization"] = "bitsandbytes"
                llm_kwargs["load_format"] = "bitsandbytes"
            elif self.quantization in {"awq", "bitsandbytes"}:
                llm_kwargs["quantization"] = self.quantization
                if self.quantization == "bitsandbytes":
                    llm_kwargs["load_format"] = "bitsandbytes"
            else:
                llm_kwargs["quantization"] = self.quantization

        self._SamplingParams = SamplingParams
        self._lora_request = None
        if self.adapter_path:
            if LoRARequest is None:
                raise RuntimeError("当前 vLLM 版本不支持 LoRA Request，无法加载 adapter。")
            rank = 16
            if isinstance(self.adapter_config, dict):
                raw_rank = self.adapter_config.get("r")
                if isinstance(raw_rank, int) and raw_rank > 0:
                    rank = raw_rank
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max(8, rank)
            self._lora_request = LoRARequest(
                Path(self.adapter_path).name,
                1,
                self.adapter_path,
                base_model_name=self.base_model_path,
            )
        self._engine = LLM(**llm_kwargs)

    @property
    def token_count_method(self) -> str:
        return "tokenizer_exact"

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
        outputs = self._engine.generate(
            prompts,
            sampling_params=self._sampling_params(),
            use_tqdm=False,
            lora_request=self._lora_request,
        )
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

    def render_prompt_for_metrics(self, prompt: str) -> str:
        return str(prompt)

    def render_batch_for_metrics(self, prompts: list[str]) -> list[str]:
        return [self.render_prompt_for_metrics(prompt) for prompt in prompts]

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self._engine.get_tokenizer()
        text = _render_chat_prompt(tokenizer, messages)
        return self._generate_many([text])[0]

    def generate_chat_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        tokenizer = self._engine.get_tokenizer()
        prompts = [_render_chat_prompt(tokenizer, msgs) for msgs in messages_batch]
        return self._generate_many(prompts)

    def render_chat_for_metrics(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self._engine.get_tokenizer()
        return _render_chat_prompt(tokenizer, messages)

    def render_chat_batch_for_metrics(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        tokenizer = self._engine.get_tokenizer()
        return [_render_chat_prompt(tokenizer, messages) for messages in messages_batch]

    def count_tokens(self, text: str) -> int:
        tokenizer = self._engine.get_tokenizer()
        count = _count_tokens_with_tokenizer(tokenizer, str(text))
        if count is not None:
            return count
        return estimate_tokens_from_text(str(text))


class LlamaCppInferenceEngine:
    """基于 llama.cpp 的 GGUF 推理引擎。"""

    def __init__(
        self,
        *,
        model_path: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
        tokenizer_path: str | None = None,
        require_gpu: bool = False,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.max_model_len = int(max_model_len)
        self.trust_remote_code = bool(trust_remote_code)
        self.require_gpu = bool(require_gpu)
        self._chat_tokenizer = _load_chat_tokenizer(
            model_path=self.model_path,
            tokenizer_path=tokenizer_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.require_gpu:
            _ensure_cuda_available("llama.cpp")

        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 llama-cpp-python 依赖，无法运行 llama.cpp 推理。") from exc

        self._engine = Llama(
            model_path=self.model_path,
            n_ctx=self.max_model_len,
            n_gpu_layers=-1,
            main_gpu=0,
            verbose=False,
        )

    @property
    def token_count_method(self) -> str:
        tokenizer_exact = self._chat_tokenizer is not None or hasattr(self._engine, "tokenize")
        return "tokenizer_exact" if tokenizer_exact else "heuristic_estimate"

    def _completion(self, prompt: str) -> str:
        response = self._engine(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=0.95 if self.temperature > 0 else 1.0,
            echo=False,
        )
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("llama.cpp 输出为空。")
        return str(choices[0].get("text", "")).strip()

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串。")
        return self._completion(prompt)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts 必须是非空字符串列表。")
        return [self.generate(str(prompt)) for prompt in prompts]

    def render_prompt_for_metrics(self, prompt: str) -> str:
        return str(prompt)

    def render_batch_for_metrics(self, prompts: list[str]) -> list[str]:
        return [self.render_prompt_for_metrics(prompt) for prompt in prompts]

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        prompt = _render_chat_prompt(self._chat_tokenizer, messages)
        return self.generate(prompt)

    def generate_chat_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        return [self.generate_chat(messages) for messages in messages_batch]

    def render_chat_for_metrics(self, messages: list[dict[str, str]]) -> str:
        return _render_chat_prompt(self._chat_tokenizer, messages)

    def render_chat_batch_for_metrics(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        return [self.render_chat_for_metrics(messages) for messages in messages_batch]

    def count_tokens(self, text: str) -> int:
        normalized = str(text)
        count = _count_tokens_with_tokenizer(self._chat_tokenizer, normalized)
        if count is None:
            count = _count_tokens_with_llama_cpp_engine(self._engine, normalized)
        if count is not None:
            return count
        return estimate_tokens_from_text(normalized)


class ExLlamaV2InferenceEngine:
    """基于 ExLlamaV2 的 EXL2 推理引擎。"""

    def __init__(
        self,
        *,
        model_path: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
        tokenizer_path: str | None = None,
        require_gpu: bool = False,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.max_model_len = int(max_model_len)
        self.trust_remote_code = bool(trust_remote_code)
        self.require_gpu = bool(require_gpu)
        self._chat_tokenizer = _load_chat_tokenizer(
            model_path=self.model_path,
            tokenizer_path=tokenizer_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.require_gpu:
            _ensure_cuda_available("exllamav2")

        try:
            from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
            from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 exllamav2 依赖，无法运行 ExLlamaV2 推理。") from exc

        config = ExLlamaV2Config()
        config.model_dir = self.model_path
        config.prepare()
        if hasattr(config, "max_seq_len"):
            config.max_seq_len = self.max_model_len
        self._config = config
        self._model = ExLlamaV2(config)
        self._cache = ExLlamaV2Cache(self._model, lazy=True)
        try:
            self._model.load_autosplit(self._cache, progress=False)
        except TypeError:
            self._model.load_autosplit(self._cache)
        self._tokenizer = ExLlamaV2Tokenizer(config)
        self._generator = ExLlamaV2BaseGenerator(self._model, self._cache, self._tokenizer)
        self._sampler_settings = ExLlamaV2Sampler.Settings()

    @property
    def token_count_method(self) -> str:
        return "tokenizer_exact"

    def _make_settings(self) -> Any:
        settings = self._sampler_settings.clone() if hasattr(self._sampler_settings, "clone") else self._sampler_settings
        settings.temperature = self.temperature
        settings.top_p = 0.95 if self.temperature > 0 else 1.0
        settings.token_repetition_penalty = 1.0
        return settings

    def _strip_prompt_prefix(self, prompt: str, text: str) -> str:
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        return text.strip()

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串。")
        result = self._generator.generate_simple(
            prompt,
            self._make_settings(),
            self.max_new_tokens,
            add_bos=True,
        )
        return self._strip_prompt_prefix(prompt, str(result))

    def generate_batch(self, prompts: list[str]) -> list[str]:
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts 必须是非空字符串列表。")
        return [self.generate(str(prompt)) for prompt in prompts]

    def render_prompt_for_metrics(self, prompt: str) -> str:
        return str(prompt)

    def render_batch_for_metrics(self, prompts: list[str]) -> list[str]:
        return [self.render_prompt_for_metrics(prompt) for prompt in prompts]

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        prompt = _render_chat_prompt(self._chat_tokenizer, messages)
        return self.generate(prompt)

    def generate_chat_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        return [self.generate_chat(messages) for messages in messages_batch]

    def render_chat_for_metrics(self, messages: list[dict[str, str]]) -> str:
        return _render_chat_prompt(self._chat_tokenizer, messages)

    def render_chat_batch_for_metrics(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch 必须是非空列表。")
        return [self.render_chat_for_metrics(messages) for messages in messages_batch]

    def count_tokens(self, text: str) -> int:
        normalized = str(text)
        count = _count_tokens_with_tokenizer(self._tokenizer, normalized)
        if count is None:
            count = _count_tokens_with_tokenizer(self._chat_tokenizer, normalized)
        if count is not None:
            return count
        return estimate_tokens_from_text(normalized)


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

    common_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "max_new_tokens": int(config.get("max_new_tokens", 256)),
        "temperature": float(config.get("temperature", 0.0)),
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        "tokenizer_path": config.get("tokenizer_path"),
        "require_gpu": bool(config.get("require_gpu", False)),
    }

    if backend == "transformers":
        return HFInferenceEngine(
            quantization=config.get("quantization"),
            use_flash_attention=bool(config.get("use_flash_attention", False)),
            **common_kwargs,
        )

    if backend == "vllm":
        return VLLMInferenceEngine(
            quantization=config.get("quantization"),
            max_model_len=int(config.get("max_model_len", 4096)),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
            **common_kwargs,
        )

    if backend == "llama.cpp":
        return LlamaCppInferenceEngine(
            max_model_len=int(config.get("max_model_len", 4096)),
            **common_kwargs,
        )

    if backend == "exllamav2":
        return ExLlamaV2InferenceEngine(
            max_model_len=int(config.get("max_model_len", 4096)),
            **common_kwargs,
        )

    raise ValueError(
        f"不支持的 backend={backend!r}。"
        " 可选值: 'transformers', 'vllm', 'llama.cpp', 'exllamav2'"
    )
