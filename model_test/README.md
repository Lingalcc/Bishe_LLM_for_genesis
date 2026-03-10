# 测试

本目录专注于第三阶段：模型与系统测试。  
分为两类：准确率评测 + 回归脚本测试。

## 统一配置

默认读取统一配置文件：`config/pipeline_config.json`  
对应配置段：

- `test.accuracy_eval`
- `test.regression`

命令行参数会覆盖配置文件中的同名参数。

## 前置依赖

1. 准确率在线评测需要可用的 API Key（建议在 `config/pipeline_config.json` 的 `test.accuracy_eval.api_key` 直接配置）
2. 回归测试依赖 Genesis 运行环境；如缺少 `gstaichi`，涉及仿真的脚本会失败

## 目录结构

- `run_accuracy_eval.py`：评估 NL->JSON 的工具调用准确率
- `run_regression_tests.py`：运行管理器/控制器/基础回归脚本

## 1) 准确率评测

在线评测（默认走 API）：

```bash
python model_test/run_accuracy_eval.py \
  --dataset-file data_prepare/genesis_franka_toolcall_alpaca.json \
  --num-samples 200 \
  --report-file model_test/accuracy_report.json
```

离线评测（使用已有预测文件）：

```bash
python model_test/run_accuracy_eval.py \
  --dataset-file data_prepare/genesis_franka_toolcall_alpaca.json \
  --predictions-file your_predictions.json \
  --num-samples 200
```

## 2) 回归脚本测试

运行全部：

```bash
python model_test/run_regression_tests.py --target all
```

单项运行：

```bash
python model_test/run_regression_tests.py --target manager
python model_test/run_regression_tests.py --target controller
python model_test/run_regression_tests.py --target basic
```

## 输出说明

- 准确率评测会生成 JSON 报告，包含：
  - `exact_match_rate`
  - `action_match_rate`
  - 逐条样本详情
- 回归测试会直接输出每个脚本是否通过，失败时终止并返回非零退出码。
