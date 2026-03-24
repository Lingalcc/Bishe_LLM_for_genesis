# 实验实施与结果汇总

更新时间：2026-03-24（已补充 Exp6 Prompt 消融结果）

## 统计口径

- 统计范围：`experiments/01_data_exp` 到 `experiments/11_exp7_prefix`，以及与实验直接对应的 `data_prepare/` 数据产物。
- 判定标准：
  - 已完成并有结构化结果：仓库内存在 `csv/json/md/png` 等可直接引用的结果文件。
  - 已执行但结果不完整：存在部分训练报告或原始日志，但缺少完整对比结果。
  - 已实现但未见结果：存在脚本、配置和说明文档，但当前仓库内没有对应报告产物。
- 备注：`experiments/finetune_exp/` 当前更像历史说明目录，不单独计入实验编号统计。

## 总览表

| 实验编号 | 实验名称 | 当前状态 | 主要结果文件 | 一句话结论 |
| --- | --- | --- | --- | --- |
| 01 | 数据生成实验 | 已完成并有结构化结果 | `data_prepare/genesis_franka_toolcall_stats.json` | 已成功生成 2000 条数据，并完成 1600/200/200 切分。 |
| 02 | 微调与对比基准 | 已执行但结果不完整 | `experiments/02_finetune_exp/reports/benchmark_report_lora.json` | 当前可直接引用的是 LoRA 训练过程指标，完整对比评测结果尚不完整。 |
| 03 | 准确率评测与 benchmark | 已实现但未见结果 | 暂无 | 评测脚本和配置已齐备，但仓库内未保留本目录下的最终报告。 |
| 04 | 仿真端到端实验 | 已实现但未见结果 | 暂无 | 端到端仿真链路已落地，但未见持久化实验数据。 |
| 05 | Exp1 数据规模对比 | 已完成并有结构化结果 | `experiments/05_exp1_data_scale/reports/*` | 数据量整体增大能提升效果，1600 样本最佳。 |
| 06 | Exp2 LoRA Rank 对比 | 已完成并有结构化结果 | `experiments/06_exp2_lora_rank/reports/*` | 在固定 600 条样本下，`rank=4` 的综合性价比最好。 |
| 07 | Exp3 微调方法对比 | 已完成并有结构化结果 | `experiments/07_exp3_methods/reports/*` | DoRA 准确率最好，但训练代价很高；QLoRA 显存最省。 |
| 08 | Exp4 量化推理对比 | 已完成并有结构化结果 | `experiments/08_exp4_inference/reports/*` | 本次结果中 `16bit` 最快，`4bit` 最省显存，`8bit` 表现最弱。 |
| 09 | Exp5 推理引擎速度基准 | 已完成并已修订口径 | `experiments/09_exp5_engine/reports/*` | 当前仅统计速度与资源指标，并强制 GPU-only；现有结果显示 `Transformers_BNB_4bit` 端到端时延低于 `LlamaCPP_GGUF_Q4_K_M`，`ExLlamaV2` 已纳入统一基准框架。 |
| 10 | Exp6 Prompt 消融 | 已完成并有结构化结果 | `experiments/10_exp6_prompt/reports/*` | 在正式测试集抽样 100 条的口径下，当前 Baseline Prompt 明显优于已尝试的 Optimized Prompt。 |
| 11 | Exp7 Prefix Caching 对照 | 已实现但未见结果 | 暂无 | 已新增开关对照脚本，可直接比较 Prefix Cache 开/关下的延迟、吞吐与显存指标。 |

## 01 数据生成实验

实验目标：生成 Franka 机械臂 `自然语言指令 -> JSON action` 数据集，并切分训练集、验证集、测试集。

当前结果来自：

- `data_prepare/genesis_franka_toolcall_stats.json`
- `data_prepare/splits/split_metadata.json`

关键数据：

- 总样本数：`2000 / 2000`
- API 调用次数：`290`
- 无效样本丢弃：`32`
- 重复样本丢弃：`195`
- 去重轮数：`7`
- 难度分布：`simple=509`、`medium=897`、`complex=594`
- `STATE_CONTEXT` 占比：`0.7`
- 数据切分：`train=1600`、`val=200`、`test=200`
- 数据集重叠检查：`train_val=0`、`train_test=0`、`val_test=0`

结论：

- 数据生成目标已经完成，且切分结果规范，没有训练集与测试集重叠问题。
- 目前后续所有训练、评测实验基本都建立在这份 `2000` 条数据及其切分之上。

## 02 微调与对比基准

实验目标：基于 LLaMA-Factory 完成 SFT 微调，并做微调前后对比。

当前仓库内可直接引用的结果文件：

- `experiments/02_finetune_exp/reports/benchmark_report_lora.json`
- `experiments/02_finetune_exp/reports/qlora_report.txt`

其中结构化结果主要来自 LoRA 训练报告：

- 训练时间：`2164.12 s`，约 `36.07 min`
- 总步数：`190`
- 训练轮数：`5.0`
- 最终损失：`0.0256`
- 最低损失：`0.0228 @ step 180`
- 峰值显存：`7929 MB`
- 平均显存：`7872 MB`

当前状态判断：

- LoRA 训练过程结果是完整的。
- `qlora_report.txt` 目前更像原始训练日志，不适合直接作为论文中的结构化结果表。
- 当前 `reports/` 下没有看到“微调前 vs 微调后准确率”这种完整结构化对比摘要，因此本实验记为“已执行但结果不完整”更稳妥。

## 03 准确率评测与推理 benchmark

实验目标：提供统一准确率评测与推理性能基准能力。

当前状态：

- 已有脚本：`experiments/03_eval_exp/run_accuracy.py`
- 已有配置：`experiments/03_eval_exp/configs/accuracy.yaml`
- 当前仓库内未发现 `experiments/03_eval_exp/reports/` 下的最终结果文件

结论：

- 这是项目中的基础评测能力入口，后续多个实验都复用了它。
- 但若只看当前仓库留存产物，本目录尚不能单独支撑“已有结果”的结论。

## 04 仿真端到端实验

实验目标：执行 `指令 -> 模型输出 action -> Genesis 执行 action` 的完整联调链路。

当前状态：

- 已有脚本：`experiments/04_sim_exp/run_e2e_sim.py`
- 已有交互环境测试：`experiments/04_sim_exp/test_genesis_interactive_env.py`
- 当前目录下未见 `reports/`、日志归档或结果数据文件

结论：

- 仿真链路已经实现，适合演示“系统可运行”。
- 但从实验管理角度看，当前缺少可直接复用的结果沉淀，后续可以补充成功率、执行耗时、轨迹截图等记录。

## 05 Exp1 数据规模对比

实验目标：研究训练数据量对微调效果的影响。

结果文件：

- `experiments/05_exp1_data_scale/reports/exp1_data_scale_results.csv`
- `experiments/05_exp1_data_scale/reports/accuracy_report_exp1_data_scale.json`
- `experiments/05_exp1_data_scale/reports/exp1_data_scale_chart.png`

汇总结果：

| 训练样本数 | Exact Match Rate | Action Match Rate |
| ---: | ---: | ---: |
| 200 | 0.290 | 0.530 |
| 400 | 0.340 | 0.605 |
| 600 | 0.330 | 0.665 |
| 800 | 0.350 | 0.660 |
| 1000 | 0.375 | 0.680 |
| 1200 | 0.355 | 0.660 |
| 1400 | 0.370 | 0.705 |
| 1600 | 0.395 | 0.705 |

补充说明：

- 当前 `accuracy_report_exp1_data_scale.json` 对应的是 `1600` 样本这一轮结果。
- 该轮详细指标：
  - `parse_ok_rate=0.995`
  - `exact_match_rate=0.395`
  - `action_match_rate=0.705`
  - `avg_latency_sec=3.1221`
  - `avg_throughput_tps=25.9309`
  - `max_peak_vram_mb=6198`

结论：

- 数据规模整体增大时，模型效果总体上升，但不是严格单调。
- 本轮实验的最佳结果出现在 `1600` 样本，说明当前数据规模尚未明显到达饱和。

## 06 Exp2 LoRA Rank 对比

实验目标：在固定 `600` 条训练样本下，比较不同 LoRA rank 的效果与成本。

结果文件：

- `experiments/06_exp2_lora_rank/reports/exp2_lora_rank_results.csv`
- `experiments/06_exp2_lora_rank/reports/accuracy_report_rank_<rank>.json`
- `experiments/06_exp2_lora_rank/reports/exp2_lora_rank_dashboard.png`

汇总结果：

| Rank | Parse OK | Exact Match | Action Match | 平均延迟(s) | 训练耗时(min) | 评测峰值显存(MB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.995 | 0.355 | 0.680 | 3.3611 | 69.20 | 6068 |
| 8 | 0.985 | 0.320 | 0.665 | 3.1926 | 34.52 | 6112 |
| 16 | 0.995 | 0.325 | 0.650 | 3.2546 | 31.87 | 6198 |
| 32 | 0.995 | 0.335 | 0.655 | 3.2210 | 50.23 | 6244 |
| 64 | 0.995 | 0.330 | 0.635 | 3.1701 | 224.17 | 6616 |

结论：

- 从准确率看，`rank=4` 最好，`exact_match_rate=0.355`，`action_match_rate=0.680`。
- 从训练成本看，`rank=64` 代价最高，训练时间达到 `224.17 min`，但精度没有同步提升。
- 如果强调综合性价比，当前结果最推荐 `rank=4`。

## 07 Exp3 微调方法对比

实验目标：比较 `LoRA / QLoRA / DoRA / GaLore` 四种微调方法在当前任务上的表现。

结果文件：

- `experiments/07_exp3_methods/reports/exp3_methods_comparison.csv`
- `experiments/07_exp3_methods/reports/accuracy_report_*.json`
- `experiments/07_exp3_methods/reports/exp3_methods_comparison_dual_axis.png`
- `experiments/07_exp3_methods/reports/exp3_methods_comparison_radar.png`

汇总结果：

| 方法 | 训练峰值显存(MB) | 训练耗时(s) | Final Loss | Parse OK | Exact Match | Action Match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 7881 | 730 | 0.0711 | 1.000 | 0.360 | 0.620 |
| QLoRA | 5274 | 598 | 0.0707 | 0.930 | 0.335 | 0.610 |
| DoRA | 7848 | 6365 | 0.0713 | 1.000 | 0.375 | 0.650 |
| GaLore | 7941 | 10137 | 0.1352 | 0.845 | 0.195 | 0.270 |

补充观察：

- LoRA 评测平均延迟约 `3.1824 s`
- QLoRA 评测平均延迟约 `2.9544 s`
- DoRA 评测平均延迟约 `23.6515 s`
- GaLore 评测平均延迟约 `2.0287 s`

结论：

- 准确率最好的是 `DoRA`，`exact_match_rate=0.375`，`action_match_rate=0.650`。
- 显存最省、训练最快的是 `QLoRA`，但准确率略低于 LoRA 与 DoRA。
- `GaLore` 在当前任务上表现明显落后，说明该方法在本项目配置下尚不稳定。
- 如果强调最终精度可优先考虑 `DoRA`；如果强调资源约束与部署可用性，`QLoRA` 更实用。

## 08 Exp4 量化推理对比

实验目标：比较 `4bit / 8bit / 16bit` 三种量化设置的推理性能与资源占用。

结果文件：

- `experiments/08_exp4_inference/reports/exp4_inference_summary.md`
- `experiments/08_exp4_inference/reports/exp4_inference_summary.json`
- `experiments/08_exp4_inference/reports/exp4_inference_results.csv`
- `experiments/08_exp4_inference/reports/exp4_inference_dashboard.png`

汇总结果：

| 量化 | 平均延迟(s) | P95延迟(s) | 样本吞吐(samples/s) | Token吞吐(tokens/s) | 峰值显存(MB) | 加载峰值显存(MB) | 相对16bit延迟加速比 | 相对16bit显存节省 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | 5.0818 | 5.9655 | 0.197 | 37.798 | 3004.0 | 2996.0 | 0.777x | 50.4% |
| 8bit | 14.0882 | 16.0195 | 0.071 | 14.013 | 3404.0 | 3394.0 | 0.280x | 43.8% |
| 16bit | 3.9464 | 4.4578 | 0.253 | 49.876 | 6052.0 | 6044.0 | 1.000x | 0.0% |

结论：

- 本次实验里 `16bit` 反而是最快的。
- `4bit` 的显存优势非常明显，峰值显存相对 `16bit` 节省约 `50.4%`，但速度略慢。
- `8bit` 在当前环境下没有体现出应有优势，速度和吞吐都落后于 `4bit` 与 `16bit`。

## 09 Exp5 推理引擎速度基准

实验目标：在 `8GB VRAM` 约束下，比较不同本地部署栈的端到端速度与资源占用。

修订后的实验口径：

- 只统计速度与资源指标，不再统计准确率；
- 所有方案都按 GPU-only 标准执行；
- 不把结果写成“同构量化下的纯引擎优劣”；
- 当前结果只解释为“部署栈整体表现”。

结果文件：

- `experiments/09_exp5_engine/reports/Transformers_4bit_benchmark.json`
- `experiments/09_exp5_engine/reports/LlamaCPP_GGUF_Q4_benchmark.json`
- `experiments/09_exp5_engine/reports/ExLlamaV2_EXL2_LocalAsset_benchmark.json`
- `experiments/09_exp5_engine/reports/exp5_speed_report.md`

当前可用结果：

| 方案 | Avg Latency(s) | P50(s) | P95(s) | Sample Throughput(samples/s) | Avg RSS(MB) | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_BNB_4bit | 5.5530 | 5.7369 | 6.6024 | 0.1801 | 1788.28 | 当前部署栈下平均时延更低 |
| LlamaCPP_GGUF_Q4_K_M | 7.0241 | 7.2612 | 8.7073 | 0.1424 | 4272.85 | 当前部署栈下尾部时延更高 |

补充说明：

- `Transformers_BNB_4bit` 对应 `bitsandbytes 4bit` 运行时量化；
- `LlamaCPP_GGUF_Q4_K_M` 对应 `GGUF Q4_K_M` 量化模型；
- `ExLlamaV2_EXL2_LocalAsset` 已纳入同一套 GPU-only benchmark，但当前本地 EXL2 资产 README 标注 `Bits 8.0`；
- 因此当前结果不适合直接写成“某引擎在严格同构 4bit 条件下一定更快或更慢”。

结论：

- 在当前仓库现有资产与默认运行参数下，`Transformers_BNB_4bit` 的端到端时延与尾部时延都优于 `LlamaCPP_GGUF_Q4_K_M`。
- 这个结论只能代表“当前部署栈 + 当前参数”的整体表现，不能直接外推为引擎上限差异。

## 10 Exp6 Prompt 消融

实验目标：比较 Baseline Prompt 与 Optimized Prompt 在空间推理和工具调用上的差异。

结果文件：

- `experiments/10_exp6_prompt/reports/baseline_accuracy.json`
- `experiments/10_exp6_prompt/reports/optimized_accuracy.json`
- `experiments/10_exp6_prompt/reports/prompt_ablation_summary.json`
- `experiments/10_exp6_prompt/reports/prompt_ablation_summary.md`

当前实验口径：

- 默认测试集已从手工 7 条高难样本切换为 `data_prepare/splits/test.json`
- 运行时按固定随机种子 `seed=42` 从正式测试集抽样 `100` 条
- 模型与推理参数保持一致：`output/qwen2.5-3b-genesis-lora-rank-4`、`local + transformers`

当前可直接引用的对比结果：

| Variant | Parse OK | Exact Match | Action Match | 平均延迟(s) | 平均吞吐(tokens/s) | 峰值显存(MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.000 | 0.370 | 0.690 | 4.8227 | 17.2702 | 6082 |
| Optimized（当前留存版本） | 0.940 | 0.290 | 0.560 | 19.0999 | 4.7898 | 6294 |

补充说明：

- 当前 `prompt_ablation_summary.json` 只保留了最新一次单独重跑的 `optimized` 结果；但 `baseline_accuracy.json` 仍保留了同一测试集、同一随机种子的基线报告，因此两者仍可用于横向比较。
- 从 `baseline_accuracy.json` 与 `optimized_accuracy.json` 看，Prompt 优化并未在通用测试集上带来收益，反而在 `exact_match_rate`、`action_match_rate`、延迟和显存上都出现退化。
- 这说明先前偏重空间关系规则、few-shot 和固定模板的 Prompt 设计更适合小规模定向样本，不一定能迁移到混合分布的正式测试集。

结论：

- Exp6 已经完成并产出结构化结果。
- 在当前“正式测试集抽样 100 条”的口径下，Baseline Prompt 明显优于当前保留的 Optimized Prompt。
- 当前证据表明，Prompt 工程如果过度强调空间规则和模板，可能会削弱模型在通用控制任务上的泛化能力。
- 后续更合理的方向不是继续无约束堆叠规则，而是将评测切分为“空间关系子集”和“通用控制子集”，分别观察 Prompt 收益。

## 当前阶段建议

如果后续要继续完善实验资产，建议优先做三件事：

1. 为 `03_eval_exp`、`04_sim_exp` 补齐标准化 `reports/` 产物，减少“能力已实现但结果未沉淀”的目录。
2. 为 `02_finetune_exp` 增加“微调前后准确率对比”的结构化摘要，避免只有训练日志。
3. 在 `10_exp6_prompt` 中把评测拆分为“空间关系子集”和“通用控制子集”，避免单一 Prompt 在混合分布上掩盖真实收益。
