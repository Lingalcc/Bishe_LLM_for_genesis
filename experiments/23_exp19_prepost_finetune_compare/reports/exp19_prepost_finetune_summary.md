# Exp19 微调前后模型对比实验

- 实验问题：微调前基座模型与多个微调后模型在结构化动作生成准确率和推理性能上的差异。
- 评价指标：`parse_ok_rate`、`exact_match_rate`、`action_match_rate`、`avg_latency_sec`、`avg_throughput_tps`、显存占用。
- 所有已有报告均来自 `experiments/*/reports/*accuracy*.json`，未重新训练模型。

## 微调前基座模型

- 模型：`model/Qwen_Qwen2.5-3B-Instruct`
- Exact Match：`0.0000`
- Action Match：`0.0350`
- 平均延迟：`2.928 s`

## 对比结果

| Model | Stage | Family | Samples | Parse OK | Exact Match | Action Match | Latency(s) | Throughput(tokens/s) | VRAM(MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrain Base | pre_finetune | base | 200 | 0.0500 | 0.0000 | 0.0350 | 2.928 | 25.503 | 6054.580 |
| LoRA Full r4 | post_finetune | lora_rank | 200 | 1.0000 | 0.3000 | 0.5850 | 4.457 | 19.573 | 6082.000 |
| LoRA r8 | post_finetune | lora_rank | 200 | 0.9850 | 0.3200 | 0.6650 | 3.193 | 26.021 | 6112.000 |
| LoRA r16 | post_finetune | lora_rank | 200 | 0.9950 | 0.3250 | 0.6500 | 3.255 | 25.762 | 6198.000 |
| LoRA r32 | post_finetune | lora_rank | 200 | 0.9950 | 0.3350 | 0.6550 | 3.221 | 25.490 | 6244.000 |
| LoRA r64 | post_finetune | lora_rank | 200 | 0.9950 | 0.3300 | 0.6350 | 3.170 | 26.099 | 6616.000 |
| Top18 r8 | post_finetune | layer_select | 200 | 0.9950 | 0.3050 | 0.5800 | 3.373 | 24.732 | 6082.000 |
| Top24 r8 | post_finetune | layer_select | 200 | 0.9900 | 0.3100 | 0.5900 | 3.688 | 22.820 | 6102.000 |
| Top28 r8 | post_finetune | layer_select | 200 | 0.9900 | 0.3100 | 0.5900 | 3.774 | 21.685 | 6114.000 |
| QLoRA | post_finetune | method | 200 | 0.9300 | 0.3350 | 0.6100 | 2.954 | 26.592 | 6068.000 |
| DoRA | post_finetune | method | 200 | 1.0000 | 0.3750 | 0.6500 | 23.651 | 3.422 | 6392.570 |
| GaLore | post_finetune | method | 200 | 0.8450 | 0.1950 | 0.2700 | 2.029 | 31.607 | 6045.140 |

## 最佳 Exact Match

- 模型：`DoRA`
- 指标值：`0.375`

## 最佳 Action Match

- 模型：`LoRA r8`
- 指标值：`0.665`

## 最低平均延迟

- 模型：`GaLore`
- 指标值：`2.029`

## 最高吞吐

- 模型：`GaLore`
- 指标值：`31.607`
