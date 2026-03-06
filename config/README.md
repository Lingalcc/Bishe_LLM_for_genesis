# 统一配置说明

统一配置文件路径：

- `config/pipeline_config.json`

该文件为四个独立目录提供默认参数：

1. `dataset_prepare`
2. `finetune`
3. `test`
4. `app`

## 覆盖优先级

1. 命令行参数
2. `config/pipeline_config.json` 中的默认值
3. 各底层脚本自身默认值

## 注意事项

1. `微调/run_finetune.py` 的 `--config` 是“统一配置文件路径”。  
如需覆盖训练 YAML，请使用透传参数：

```bash
python 微调/run_finetune.py -- --config LLaMA-Factory/examples/train_lora/qwen3_lora_sft_genesis_toolcall.yaml
```

2. 仿真应用可通过 `app.interactive.enabled` 一键开关启动权限。

3. API Key 推荐直接写在配置中：

- `dataset_prepare.augment.api_key`
- `test.accuracy_eval.api_key`
- `app.model.api_key`

如留空，程序才会尝试回退到 `api_key_env` 指定的环境变量。

4. 状态注入相关：

- 数据集构建比例：`dataset_prepare.generate.state_context_ratio`
- 指令到动作时注入开关：`app.state_injection.enable_instruction_to_action`
- 端到端注入开关：`app.state_injection.enable_instruction_to_motion`
