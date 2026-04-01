from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


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
