# BirdSet严格小样本链路INT8量化

本目录对BirdSet grouped 5/10/20-shot实验中按三seed平均适配分数预先选出的9组策略执行PTQ；每组保留42、123、2026三个seed，共27个TFLite模型。BirdSet test固定为197条长录音、18,265个五秒片段和91,325个一秒窗口。

所有模型均为int8输入/输出、0个浮点张量、无`QUANTIZE`/`DEQUANTIZE`算子。校准使用256个代表性窗口，在Xeno训练特征与对应BirdSet support间等额分配；三个held-out域均不参与校准或选型。

## FP32–INT8配对结果

数值为三seed均值±样本标准差。BirdSet纯单物种Macro-F1只覆盖公共test中有正样本的支持类别。

| 特征 | shot/策略 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 |
|---|---|---:|---:|---:|
| MFCC | 5 / `bn_head_replay` | 48.41±0.44% → 48.39±1.59% | 20.06±0.57% → 20.15±0.82% | 13.32±0.56% → 14.45±1.02% |
| MFCC | 10 / `bn_head_replay` | 47.89±0.61% → 47.81±0.71% | 19.88±0.29% → 20.69±0.57% | 13.44±0.25% → 15.20±0.43% |
| **MFCC** | **20 / `bn_head_replay`** | 48.53±0.40% → 47.72±0.95% | 23.89±0.44% → **24.12±1.43%** | 17.57±0.49% → **18.40±1.35%** |
| LogMel | 5 / `bn_head_replay` | 56.10±0.63% → 53.14±1.57% | 20.31±2.17% → 20.22±5.03% | 13.32±2.35% → 13.20±1.49% |
| LogMel | 10 / `bn_head_replay` | 56.11±1.31% → 54.40±1.88% | 20.39±2.02% → 23.09±4.19% | 13.66±2.57% → 14.19±0.66% |
| LogMel | 20 / `head_only` | 57.81±1.24% → **55.62±2.18%** | 18.66±0.91% → 21.18±1.76% | 10.62±0.75% → 12.21±1.24% |
| PCEN | 5 / `bn_head` | 55.27±1.34% → 17.97±2.16% | 36.11±1.11% → 19.78±0.09% | 27.11±0.40% → 4.25±0.08% |
| PCEN | 10 / `bn_head` | 54.76±1.29% → 19.03±1.08% | **37.56±2.08%** → 19.84±0.04% | **28.06±1.60%** → 4.34±0.11% |
| PCEN | 20 / `bn_head` | 57.85±0.06% → 16.85±1.33% | 29.68±0.41% → 19.28±0.18% | 23.42±0.18% → 4.18±0.05% |

FP32目标域最优是PCEN 10-shot `bn_head`，但PCEN PTQ严重失稳。若只强调BirdSet量化后目标指标，推荐MFCC 20-shot `bn_head_replay`；若同时强调Xeno和DB3V绝对精度，LogMel 20-shot `head_only`更均衡，其量化后Xeno Macro-F1为55.62%±2.18%，完整DB3V的原始来源录音级Macro-F1为67.34%。

`summary.csv`保存27个逐seed配对，`aggregate.csv`保存9组均值和标准差，`models/<chain_id>/evaluation.json`保存Xeno、BirdSet和DB3V逐粒度结果。所有DB3V主指标已刷新为1,363条原始来源录音级；旧八秒块级数值仍只作为次级粒度保存在原始报告中。

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families birdset_strict_fewshot `
  --output-dir src\experiments\BirdSet_strict_INT8_quantization_8class `
  --representative-samples 256 --batch-size 128 --num-threads 4
```
