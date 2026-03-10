from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.local_llm_engine import LocalLLMEngine


def main() -> None:
    engine = LocalLLMEngine(
        model_path=str(REPO_ROOT / "model" / "Qwen_Qwen2.5-7B-Instruct"),
        quantization="4bit",  # 可选: "awq", "4bit", "8bit", None
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    print("model initialized with backend:", engine.backend)

    prompt = "把夹爪打开，移动到 [0.65, 0.0, 0.20]，然后闭合夹爪。只输出 JSON。"
    json_text = engine.generate(prompt, temperature=0.1)
    print(json_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
