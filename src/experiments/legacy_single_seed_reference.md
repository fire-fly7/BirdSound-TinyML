# 已淘汰单种子链路参考

本文件只保留已淘汰实验中对当前严格链路仍有解释价值的结论，不是训练、选型、
量化或部署输入。旧模型权重、逐轮历史和执行入口已从当前树移除。

旧协议使用随机种子42和`head`、`last_block`、`all`策略。其中`head`实际训练两个
Dense层，`all`仍冻结BatchNorm，因此不能分别视为当前的严格Head-Only和
Full Fine-Tuning；BirdSet旧实验还只覆盖grouped 5-shot。

## 迁移保留的关键结论

| 旧场景 | 旧选中链路 | Xeno Macro-F1 | 目标域指标 | 当前替代 |
|---|---|---:|---:|---|
| DB3V综合FP32 | LogMel 20-shot `all` | 63.45% | DB3V Macro-F1 68.86% | 严格三seed按适配分数选型 |
| DB3V目标域FP32最高 | PCEN 20-shot `last_block` | 61.51% | DB3V Macro-F1 70.60% | PCEN 20-shot `full` 70.50%±0.40% |
| DB3V旧INT8推荐 | LogMel 10-shot `last_block` | 58.54% | DB3V Macro-F1 69.70% | LogMel 10-shot `head_only` 68.14%±0.17% |
| BirdSet旧FP32目标域 | PCEN 5-shot `head` | 59.44% | Top-1 35.70%，纯单物种F1 24.66% | PCEN 10-shot `bn_head`三seed结果 |
| BirdSet旧INT8折中 | LogMel 5-shot `head` | 53.68% | Top-1 23.80%，DB3V F1 58.08% | 当前BirdSet严格INT8选型 |

旧结果证明了三个后来被严格协议确认的现象：LogMel具有较稳定的跨域表现，PCEN在
FP32目标域适配中可能更高，但直接PTQ容易严重退化；单一随机种子不足以支持部署
推荐。当前正式结果以三个随机种子、公共held-out、Xeno遗忘测试、跨域复测和严格
INT8结果为准。

## 被移除汇总的来源指纹

| 原文件 | SHA-256 |
|---|---|
| `DB3V_fewshot_comparison_8class/comparison_summary.csv` | `D828BEB6555F7D49551F0690D853358956AB3AC604E5A40CF4B8613CCC6B7576` |
| `BirdSet_fewshot_8class/comparison_summary.csv` | `7A120F3575888148E6B05A1316F82E43442791E36E1D0B61E820B56FEA6F9B3A` |
| `BirdSet_fewshot_8class/experiment_protocol.json` | `A7D9A272AE5E6743DA19AAEAFC4A31A0CA8DAA876BC2F8838AFCD75EA389D7DC` |
| `INT8_quantization_8class/summary.csv` | `C75B23F29D94EC1C3A229B4F5EF8561A0AA1CB7F590DBAECDD834C1D68E754A3` |
| `INT8_quantization_8class/experiment_protocol.json` | `91DB1F37645A16F7CE8B5F04EF1CB2597180C0967937E58DE9CC02D782D14346` |

当前替代结果位于：

- `DB3V_fewshot_ablation_multiseed_8class/`
- `BirdSet_fewshot_ablation_multiseed_8class/`
- `ZeroShot_strict_INT8_quantization_8class/`
- `DB3V_strict_INT8_quantization_8class/`
- `BirdSet_strict_INT8_quantization_8class/`
