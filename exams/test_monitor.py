from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.local_llm_engine import LocalLLMEngine
from src.eval.performance_monitor import monitor_inference_performance
prompt = "把夹爪打开，移动到 [0.65, 0.0, 0.20]，然后闭合夹爪。只输出 JSON"
model = LocalLLMEngine(
        model_path=str(REPO_ROOT / "model" / "Qwen_Qwen2.5-7B-Instruct"),
        quantization="4bit",  # 可选: "awq", "4bit", "8bit", None
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
print("model initialized with backend:", model.backend)

with monitor_inference_performance(input_text=prompt) as mon:
    output = model.generate(prompt)
    mon.set_output_text(output)

print(mon.metrics)
# {'peak_vram_mb': ..., 'latency_sec': ..., 'throughput_tps': ...}
