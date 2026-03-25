# 实验 11_exp7_adarank 使用说明

本实验用于比较两种参数高效微调策略在机械臂 ToolCall 生成任务上的效果：

- `AdaLoRA`：动态自适应 rank 分配。
- `Static Rank + Block-wise Freezing`：先做层敏感度打分，再仅对高敏感层施加更高 rank 的 LoRA。

实验目录：[`experiments/09_adaptive_vs_static`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static)


## 1. 实验目标

本实验希望回答一个具体问题：

在机器人控制指令生成任务中，与其让 AdaLoRA 在全层范围内动态分配低秩预算，是否可以先通过梯度敏感度打分识别关键层，再把固定参数预算集中投放到高价值层，从而得到更好的 JSON / ToolCall 格式稳定性与动作语义准确率。

当前实验设计包含两条路线：

- 基线 1：`AdaLoRA`
  使用全层目标模块 `all`，初始 rank 更高，训练过程中动态收缩与重分配。
- 提出方法：`Static Rank`
  假设 `0-15` 层较不重要，`16-31` 层较重要，仅在高层 block 注入 LoRA，并把 rank 提高到 `32`，保持总体预算与 AdaLoRA 大致同量级。


## 2. 目录结构

```text
experiments/09_adaptive_vs_static/
├── README.md
├── run_pipeline.py
├── run_layer_scoring.py
├── configs/
│   ├── train_adalora.yaml
│   └── train_static_rank.yaml
├── logs/
├── reports/
└── .cache/
```

各文件作用如下：

- [`configs/train_adalora.yaml`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/configs/train_adalora.yaml)
  AdaLoRA 训练配置。
- [`configs/train_static_rank.yaml`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/configs/train_static_rank.yaml)
  静态层冻结 + 高 rank LoRA 配置。
- [`run_layer_scoring.py`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/run_layer_scoring.py)
  对基础模型做梯度敏感度分析，输出逐层分数。
- [`run_pipeline.py`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/run_pipeline.py)
  一键串联训练、打分、再次训练和评测流程。


## 3. 前置条件

在启动实验前，请确认以下条件满足：

- 已完成数据切分：
  需要存在 [`data_prepare/splits/train.json`](/home/lin/Bishe_LLM_for_genesis/data_prepare/splits/train.json)、[`data_prepare/splits/val.json`](/home/lin/Bishe_LLM_for_genesis/data_prepare/splits/val.json)、[`data_prepare/splits/test.json`](/home/lin/Bishe_LLM_for_genesis/data_prepare/splits/test.json)
- 已准备基础模型：
  默认模型路径为 [`model/Qwen_Qwen2.5-3B-Instruct`](/home/lin/Bishe_LLM_for_genesis/model/Qwen_Qwen2.5-3B-Instruct)
- 已安装训练依赖：
  至少需要 `transformers`、`torch`、`pyyaml`，训练阶段还需要当前仓库已有的 `LlamaFactory` 运行环境。
- 建议使用你当前项目已有的 conda 环境运行，避免出现 `llamafactory` 或 `trust_remote_code` 相关依赖不一致。


## 4. 单独运行每个阶段

如果你想分阶段调试，而不是一次性跑完整流程，可以按下面方式执行。

### 4.1 运行层敏感度打分

```bash
python experiments/09_adaptive_vs_static/run_layer_scoring.py \
  --model-path model/Qwen_Qwen2.5-3B-Instruct \
  --data-path data_prepare/splits/train.json \
  --output-path experiments/09_adaptive_vs_static/reports/layer_scores.json \
  --sample-size 100
```

输出文件：

- [`layer_scores.json`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/reports/layer_scores.json)

输出内容包含：

- `meta`：模型路径、样本数、设备、dtype 等元信息
- `summary`：top-5 / bottom-5 层统计
- `layers`：逐层聚合分数
- `ranking`：按分数排序后的层列表


### 4.2 单独训练 AdaLoRA

```bash
python experiments/02_finetune_exp/run_train.py \
  --base-config configs/base.yaml \
  --config experiments/09_adaptive_vs_static/.cache/train_adalora_override.yaml \
  output_dir=/home/lin/Bishe_LLM_for_genesis/output/exp11_exp7_adarank/adalora
```

注意：

- 更推荐通过 `run_pipeline.py` 自动生成 `.cache` 下的 override 配置，而不是手工维护。
- 真正的训练数据会由仓库现有训练器自动映射到运行时 split 数据集，不需要手工指定 `dataset=__train_split__`。


### 4.3 单独训练静态高层 LoRA

```bash
python experiments/02_finetune_exp/run_train.py \
  --base-config configs/base.yaml \
  --config experiments/09_adaptive_vs_static/.cache/train_static_override.yaml \
  output_dir=/home/lin/Bishe_LLM_for_genesis/output/exp11_exp7_adarank/static_rank
```


### 4.4 单独评测某个模型

```bash
python experiments/03_eval_exp/run_accuracy.py \
  --base-config configs/base.yaml \
  --config experiments/09_adaptive_vs_static/.cache/eval_adalora_override.yaml
```

或者：

```bash
python experiments/03_eval_exp/run_accuracy.py \
  --base-config configs/base.yaml \
  --config experiments/09_adaptive_vs_static/.cache/eval_static_override.yaml
```


## 5. 一键运行完整实验

这是最推荐的启动方式。

### 5.1 先做 dry-run 检查

```bash
python experiments/09_adaptive_vs_static/run_pipeline.py --dry-run --skip-eval
```

作用：

- 检查命令拼接是否正确
- 检查日志目录和缓存配置是否能正常生成
- 不真正启动训练


### 5.2 跑训练 + 打分，不做最终评测

```bash
python experiments/09_adaptive_vs_static/run_pipeline.py --skip-eval
```

适用场景：

- 先确认两组训练都能跑通
- 先拿到 `layer_scores.json`
- 评测准备稍后再做


### 5.3 跑完整流程

```bash
python experiments/09_adaptive_vs_static/run_pipeline.py
```

完整流程包含 4 步：

1. 训练 AdaLoRA 模型
2. 运行层敏感度打分脚本
3. 训练静态高层 LoRA 模型
4. 调用现有评测脚本比较两个模型

如果你想在不影响原有 `important_rank4 / high18_rank8` 设定的前提下，额外补充“筛层 + 筛模块”的二维过滤分支，可以启用：

```bash
python experiments/09_adaptive_vs_static/run_pipeline.py \
  --enable-high18-rank8-2d \
  --high-layer-2d-module-types q_proj,k_proj,v_proj,o_proj,down_proj
```

这会额外生成一个独立分支 `high18_rank8_2d`：

- 层维度：仍使用 `layer_scores` 排名后的 Top-18 层
- 模块维度：默认只保留 `q_proj,k_proj,v_proj,o_proj,down_proj`
- 兼容性：不会覆盖原有 `high18_rank8` 的配置、日志、模型输出与汇总结果


### 5.4 指定 GPU

如果你想显式指定 GPU，例如使用 `0` 号卡：

```bash
python experiments/09_adaptive_vs_static/run_pipeline.py --gpus 0
```

如果后续要尝试多卡，也可以传：

```bash
python experiments/09_adaptive_vs_static/run_pipeline.py --gpus 0,1
```


### 5.5 调整层打分样本数

默认打分样本数是 `100`。如果你想提高打分稳定性，可以改为 `200`：

```bash
python experiments/09_adaptive_vs_static/run_pipeline.py \
  --layer-sample-size 200 \
  --skip-eval
```


## 6. 实验输出位置

### 6.1 训练输出

- AdaLoRA 模型输出目录：
  [`output/exp11_exp7_adarank/adalora`](/home/lin/Bishe_LLM_for_genesis/output/exp11_exp7_adarank/adalora)
- Static Rank 模型输出目录：
  [`output/exp11_exp7_adarank/static_rank`](/home/lin/Bishe_LLM_for_genesis/output/exp11_exp7_adarank/static_rank)


### 6.2 日志输出

- AdaLoRA 训练日志：
  [`logs/train_adalora.log`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/logs/train_adalora.log)
- 层打分日志：
  [`logs/layer_scoring.log`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/logs/layer_scoring.log)
- Static Rank 训练日志：
  [`logs/train_static_rank.log`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/logs/train_static_rank.log)
- 评测日志：
  [`logs/eval_adalora.log`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/logs/eval_adalora.log)
  [`logs/eval_static_rank.log`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/logs/eval_static_rank.log)


### 6.3 报告输出

- 层敏感度报告：
  [`reports/layer_scores.json`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/reports/layer_scores.json)
- AdaLoRA 评测报告：
  [`reports/accuracy_report_adalora.json`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/reports/accuracy_report_adalora.json)
- Static Rank 评测报告：
  [`reports/accuracy_report_static_rank.json`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/reports/accuracy_report_static_rank.json)
- 二维过滤补充报告（启用 `--enable-high18-rank8-2d` 后生成）：
  [`reports/high18_rank8_2d_layers.json`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/reports/high18_rank8_2d_layers.json)
  [`reports/comparison_summary_2d.json`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/reports/comparison_summary_2d.json)
  [`reports/comparison_summary_2d.md`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/reports/comparison_summary_2d.md)


## 7. 当前静态方法的默认假设

当前版本中，静态方法使用的是一个“先验假设”：

- `0-15` 层视为较低敏感层
- `16-31` 层视为较高敏感层

因此 [`train_static_rank.yaml`](/home/lin/Bishe_LLM_for_genesis/experiments/09_adaptive_vs_static/configs/train_static_rank.yaml) 中的 `lora_target` 正则只匹配高层 block 的注意力投影和 MLP 投影。

这意味着当前版本是“半自动实验”：

- `run_layer_scoring.py` 会真实输出层分数
- 但 `train_static_rank.yaml` 还没有根据 `layer_scores.json` 自动改写层范围

如果后续你希望升级为“全自动 static rank 生成”，建议下一步做：

1. 读取 `reports/layer_scores.json`
2. 自动选出 top-k 层
3. 生成新的正则或模块白名单
4. 再启动 static LoRA 训练


## 8. 常见问题

### 8.1 为什么 YAML 里写的是 `dataset: genesis_franka_toolcall_train`，但训练时又会看到 `__train_split__`？

这是本仓库现有训练入口的设计：

- 配置文件表达的是实验意图
- 真正运行时，`run_train.py` 会把 `train.json / val.json` 注入成运行时数据集映射

这样做的好处是：

- 可以严格绑定本地切分文件
- 避免误用旧数据集名字
- 兼容仓库已有的数据泄漏检查逻辑


### 8.2 为什么静态方法把 rank 提高到 32？

因为它只对高层 block 注入 LoRA，而不是全层注入。提高单层 rank 的目的是在总体参数预算接近的前提下，把更多容量集中到高价值层。


### 8.3 层打分很慢怎么办？

可以先把样本数调小，例如：

```bash
python experiments/09_adaptive_vs_static/run_layer_scoring.py --sample-size 20
```

适合先做脚本功能验证。确认逻辑正确后，再把样本数提高到 `100` 或 `200`。


## 9. 推荐使用顺序

如果你要正式开跑，我建议按下面顺序操作：

1. 先执行 dry-run，确认命令与路径无误。
2. 再执行 `--skip-eval`，先观察两阶段训练和层打分是否稳定。
3. 确认模型输出目录正常后，再执行完整评测。
4. 最后把 `layer_scores.json` 与两份 accuracy report 做汇总分析。


## 10. 快速命令汇总

```bash
# 1) 仅检查命令，不真正训练
python experiments/09_adaptive_vs_static/run_pipeline.py --dry-run --skip-eval

# 2) 跑训练 + 层打分
python experiments/09_adaptive_vs_static/run_pipeline.py --skip-eval

# 3) 跑完整实验
python experiments/09_adaptive_vs_static/run_pipeline.py

# 4) 单独运行层敏感度打分
python experiments/09_adaptive_vs_static/run_layer_scoring.py \
  --model-path model/Qwen_Qwen2.5-3B-Instruct \
  --data-path data_prepare/splits/train.json \
  --output-path experiments/09_adaptive_vs_static/reports/layer_scores.json \
  --sample-size 100
```


## 11. 后续建议

如果你准备把这个实验写进论文或答辩材料，我建议接下来再补两项：

- 结果汇总脚本：
  自动读取两份 accuracy report 和层打分结果，输出 CSV / 图表
- 自动静态层选择：
  根据 `layer_scores.json` 自动构建 static rank 配置，避免手工假设 top-16 层

这样这组实验会从“可运行对比”进一步升级为“可复现、可分析、可扩展”的完整实验框架。
