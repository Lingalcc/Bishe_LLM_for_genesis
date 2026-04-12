from __future__ import annotations

import json
import os
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

from packaging.version import Version


@contextmanager
def temporary_quant_hf_overrides_dict(model_config: Any) -> Iterator[None]:
    """兼容 vLLM 在量化配置阶段把 callable hf_overrides 当作 dict 使用的问题。"""

    original_hf_overrides = getattr(model_config, "hf_overrides", None)
    if callable(original_hf_overrides):
        model_config.hf_overrides = {}
        try:
            yield
        finally:
            model_config.hf_overrides = original_hf_overrides
        return
    yield


def apply_vllm_transformers_config_patch() -> bool:
    """为 vLLM 0.16 与新版 transformers 的导入差异补一个最小兼容层。"""

    try:
        from transformers import configuration_utils
    except Exception:
        return False

    if hasattr(configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES"):
        return True

    legacy_value = getattr(configuration_utils, "ALLOWED_LAYER_TYPES", None)
    if legacy_value is None:
        legacy_value = set()
    setattr(configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES", legacy_value)
    return True


def get_vllm_environment_compat_error() -> str | None:
    """检查当前 vLLM 相关依赖组合是否明显不兼容。"""

    if os.environ.get("LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None

    try:
        vllm_version = metadata.version("vllm")
    except Exception:
        return None

    try:
        transformers_version = metadata.version("transformers")
    except Exception:
        transformers_version = None

    if (
        transformers_version is not None
        and Version(vllm_version) >= Version("0.16.0")
        and Version(transformers_version) < Version("4.56.0")
    ):
        return (
            "当前环境中的 vLLM / transformers 版本不兼容："
            f" vllm={vllm_version}, transformers={transformers_version}。"
            "vLLM 0.16.0 需要 transformers>=4.56.0,<5。"
            "请先升级 transformers，或降级到与当前 transformers 匹配的 vllm 版本。"
        )

    try:
        compressed_tensors_version = metadata.version("compressed-tensors")
    except Exception:
        compressed_tensors_version = None

    if (
        compressed_tensors_version is not None
        and Version(vllm_version) == Version("0.16.0")
        and Version(compressed_tensors_version) != Version("0.13.0")
    ):
        return (
            "当前环境中的 vLLM / compressed-tensors 版本不兼容："
            f" vllm={vllm_version}, compressed-tensors={compressed_tensors_version}。"
            "仓库当前锁定的组合是 vllm==0.16.0 与 compressed-tensors==0.13.0。"
            "请执行 `pip install compressed-tensors==0.13.0`，"
            "或重新按 requirements.txt 同步环境。"
        )
    return None


def ensure_vllm_environment_compatible() -> None:
    """在真正导入 vLLM 之前给出更清晰的环境错误。"""

    error = get_vllm_environment_compat_error()
    if error:
        raise RuntimeError(error)


def resolve_vllm_quantization(
    model_path: str | Path,
    requested_quantization: str | None,
) -> str | None:
    """根据模型目录内的量化配置，修正传给 vLLM 的 quantization 参数。"""

    resolved_requested = None
    if requested_quantization is not None:
        text = str(requested_quantization).strip().lower()
        resolved_requested = text or None

    try:
        config_path = Path(model_path).expanduser() / "config.json"
    except Exception:
        return resolved_requested

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return resolved_requested

    if not isinstance(payload, dict):
        return resolved_requested

    quant_cfg = payload.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        return resolved_requested

    quant_method = quant_cfg.get("quant_method")
    if not isinstance(quant_method, str) or not quant_method.strip():
        return resolved_requested

    normalized_quant_method = quant_method.strip().lower()

    # llmcompressor 导出的目录会把 quant_method 写成 compressed-tensors。
    # 如果外部仍沿用 awq 参数，这里自动纠偏，避免 vLLM 因方法名不一致报错。
    if normalized_quant_method == "compressed-tensors":
        if resolved_requested in {None, "awq", "compressed-tensors"}:
            return "compressed-tensors"

    return resolved_requested


def apply_vllm_speculative_bnb_patch() -> bool:
    """为 vLLM speculative + bitsandbytes 4bit 场景打一个幂等兼容补丁。"""

    try:
        from vllm.model_executor.model_loader import weight_utils
    except Exception:
        return False

    if getattr(weight_utils, "_llm_genesis_spec_bnb_patch_applied", False):
        return True

    original_get_quant_config = weight_utils.get_quant_config

    def patched_get_quant_config(model_config: Any, load_config: Any) -> Any:
        with temporary_quant_hf_overrides_dict(model_config):
            return original_get_quant_config(model_config, load_config)

    patched_get_quant_config.__name__ = getattr(
        original_get_quant_config,
        "__name__",
        "patched_get_quant_config",
    )
    patched_get_quant_config.__doc__ = getattr(original_get_quant_config, "__doc__", None)
    patched_get_quant_config._llm_genesis_original = original_get_quant_config

    weight_utils.get_quant_config = patched_get_quant_config
    weight_utils._llm_genesis_spec_bnb_patch_applied = True
    return True
