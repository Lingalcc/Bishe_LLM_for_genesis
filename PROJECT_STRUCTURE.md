# 项目结构（四目录独立模式）

按你的要求，流程拆分为四个互相独立的目录，不再依赖统一式调度入口。

## 统一配置文件

四个目录共享同一份配置文件：

- `config/pipeline_config.json`

你可以只改这一个文件来维护全流程默认参数。  
各入口脚本支持 `--config` 指定其他配置文件，且命令行参数优先级高于配置文件。

## 四个独立目录

- `数据集准备校准/`：生成数据、API 增强、数据合法性校准
- `微调/`：启动 LLaMA-Factory 微调
- `测试/`：准确率评测 + 回归脚本测试
- `仿真应用/`：交互式仿真执行（Franka 控制）

## 目录内入口（独立执行）

1. 数据集准备校准

```bash
python 数据集准备校准/run_generate_dataset.py --num-samples 4000 --seed 42
python 数据集准备校准/run_augment_dataset.py --num-source 800 --aug-per-sample 2 --model gpt-5
python 数据集准备校准/run_dataset_calibration.py --dataset-file data_prepare/genesis_franka_toolcall_alpaca.json
```

2. 微调

```bash
python 微调/run_finetune.py --dry-run --gpus 0
python 微调/run_finetune.py --gpus 0
```

3. 测试

```bash
python 测试/run_accuracy_eval.py --num-samples 200
python 测试/run_regression_tests.py --target all
```

4. 仿真应用

```bash
python 仿真应用/run_instruction_to_action.py
python 仿真应用/run_action_to_motion.py
python 仿真应用/run_instruction_to_motion.py
```

## 底层实现说明

四目录入口脚本仅做薄封装，核心逻辑仍在 `pipeline/` 下：

- `pipeline/dataset_prepare/`
- `pipeline/finetune/`
- `pipeline/accuracy_test/`
- `pipeline/app/`

这样的组织方式兼顾了“对外独立目录清晰使用”和“对内代码复用”。
