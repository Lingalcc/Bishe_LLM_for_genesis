from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
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
    """检查当前 vLLM / transformers 组合是否明显不兼容。"""

    try:
        vllm_version = metadata.version("vllm")
        transformers_version = metadata.version("transformers")
    except Exception:
        return None

    if Version(vllm_version) >= Version("0.16.0") and Version(transformers_version) < Version("4.56.0"):
        return (
            "当前环境中的 vLLM / transformers 版本不兼容："
            f" vllm={vllm_version}, transformers={transformers_version}。"
            "vLLM 0.16.0 需要 transformers>=4.56.0,<5。"
            "请先升级 transformers，或降级到与当前 transformers 匹配的 vllm 版本。"
        )
    return None


def ensure_vllm_environment_compatible() -> None:
    """在真正导入 vLLM 之前给出更清晰的环境错误。"""

    error = get_vllm_environment_compat_error()
    if error:
        raise RuntimeError(error)


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
