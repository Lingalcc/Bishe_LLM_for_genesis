# Exp11 投机解码引擎对比报告

## 实验目标

- 将 `exp11` 的主模型、助手模型、数据集、样本数、温度、最大生成长度与预热设置对齐到 `exp8`。
- `Transformers` 侧继续使用 `assistant_model` assisted generation；`vLLM` 侧仅保留基线解码，并固定使用 `float16` 精度加载 3B 模型。
- 当前场景包括：`vLLM 无投机解码`、`Transformers 无投机解码`、`Transformers 投机解码`。

## 结果总表

| Case | Backend | Mode | Method | Precision | Avg Latency (s) | Tokens/s | Peak VRAM (MB) | Parse OK Rate | Status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| vllm_standard | vllm | 无投机解码 | none | - | 0.0000 | 0.000 | 0.0 | 0.0000 | failed |
| transformers_standard | transformers | 无投机解码 | none | - | 0.0000 | 0.000 | 0.0 | 0.0000 | failed |
| transformers_speculative | transformers | 投机解码 | assistant_model | - | 0.0000 | 0.000 | 0.0 | 0.0000 | failed |

## 口径说明

- vLLM 仅运行基线解码，主模型以 float16 精度加载，用于和 Transformers 基线与投机解码结果对比。
- 与 exp8 保持一致：BF16/FP16 自动选择、单模型生成，不启用投机解码。
- 与 exp8 保持一致：BF16/FP16 自动选择、主模型与助手模型优先同卡加载、assistant_confidence_threshold 与 constant schedule 按 exp8 配置执行。

## 后端结论

- 当前没有形成可比的成功结果对。

## 失败情况

- `vllm_standard` 失败，状态为 `failed`，原因：`RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`。
- `transformers_standard` 失败，状态为 `failed`，原因：`ValueError: Argument `logits_processor` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.`。
- `transformers_speculative` 失败，状态为 `failed`，原因：`ValueError: Argument `logits_processor` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.`。