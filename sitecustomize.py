from __future__ import annotations

try:
    from src.utils.vllm_compat import apply_vllm_speculative_bnb_patch

    apply_vllm_speculative_bnb_patch()
except Exception:
    pass
