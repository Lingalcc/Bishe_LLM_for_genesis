# finetune_exp

该目录用于维护本项目微调所需的 LLaMA-Factory 训练配置，目标是将业务侧改动与 `LlamaFactory` 源码解耦。

- 推荐修改位置：`experiments/finetune_exp/llamafactory/train_lora_sft.yaml`
- 不推荐直接修改：`LlamaFactory/examples/...`

默认微调入口会优先读取本目录下的配置文件。
