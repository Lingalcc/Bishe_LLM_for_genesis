# 实验08 Exp4 推理量化对比

- 模型：`/home/lin/Bishe_LLM_for_genesis/model/Qwen_Qwen2.5-3B-Instruct`
- 后端：`transformers`
- batch size：`1`
- 样本数：`24`
- prompts：`experiments/08_exp4_inference/reports/exp4_prompts_used.json`

| 量化 | 平均延迟(s) | P95延迟(s) | 样本吞吐(samples/s) | Token吞吐(tokens/s) | 峰值显存(MB) | 加载峰值显存(MB) | 相对16bit延迟加速比 | 相对16bit显存节省 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | 5.0818 | 5.9655 | 0.197 | 37.798 | 3004.0 | 2996.0 | 0.777x | 50.4% |
| 8bit | 14.0882 | 16.0195 | 0.071 | 14.013 | 3404.0 | 3394.0 | 0.280x | 43.8% |
| 16bit | 3.9464 | 4.4578 | 0.253 | 49.876 | 6052.0 | 6044.0 | 1.000x | 0.0% |
