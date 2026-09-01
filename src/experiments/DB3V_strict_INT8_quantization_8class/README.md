# DB3V严格小样本链路INT8量化

本目录对DB3V严格多种子实验中按平均适配分数预先选出的9组策略执行PTQ；每组保留42、123、2026三个seed，共27个TFLite模型。量化结果不参与重新选择FP32策略。

DB3V主指标统一为最大20-shot support之外的970条独立原始来源录音；对应7,660个八秒块和61,280个一秒窗口。每个模型还在480条Xeno验证录音和BirdSet公共test（197条长录音、18,265个五秒片段）上配对复测。

## 严格性与校准隔离

全部27个模型均满足int8输入、int8输出、0个浮点张量、无`QUANTIZE`/`DEQUANTIZE`算子。每个模型使用256个代表性窗口校准，Xeno训练特征与对应shot的DB3V support等额分配；Xeno验证、DB3V test和BirdSet test均不参与校准。

## FP32–INT8配对结果

数值为三seed均值±样本标准差。DB3V列是原始来源录音级Macro-F1。

| 特征 | shot/策略 | Xeno Macro-F1（FP32→INT8） | DB3V Macro-F1（FP32→INT8） |
|---|---|---:|---:|
| MFCC | 5 / `head_only` | 47.73±0.31% → 47.78±0.31% | 55.17±0.20% → 54.63±0.18% |
| MFCC | 10 / `bn_head_replay` | 49.16±0.49% → 47.01±0.50% | 57.49±0.27% → 56.17±0.63% |
| MFCC | 20 / `bn_head_replay` | 48.39±0.45% → 48.02±0.55% | 57.40±0.86% → 56.99±0.92% |
| LogMel | 5 / `head_only` | 59.59±0.23% → 53.32±0.32% | 67.23±0.48% → 63.79±0.19% |
| LogMel | 10 / `bn_head_replay` | 61.00±0.26% → 59.38±0.50% | 68.12±0.38% → 66.67±0.96% |
| **LogMel** | **20 / `bn_head_replay`** | **61.36±0.21% → 60.01±1.32%** | **68.84±0.68% → 67.73±0.59%** |
| PCEN | 5 / `full` | 62.65±0.82% → 37.15±0.50% | **71.93±0.48%** → 55.20±1.10% |
| PCEN | 10 / `full` | 62.83±0.67% → 32.34±2.42% | 71.78±1.00% → 46.09±6.09% |
| PCEN | 20 / `full` | 61.64±0.65% → 28.08±4.86% | 71.27±0.20% → 37.73±9.40% |

FP32目标域最高的是PCEN 5-shot `full`，但PCEN PTQ损失和方差都很大。当前严格INT8推荐为LogMel 20-shot `bn_head_replay`：DB3V 67.73%±0.59%，Xeno 60.01%±1.32%。

`summary.csv`保存27个逐seed配对，`aggregate.csv`保存9组均值和标准差，`models/<chain_id>/evaluation.json`保存Xeno、BirdSet、DB3V的FP32引用、INT8逐粒度结果和差值。协议与输入scale/zero-point分别见`experiment_protocol.json`和`input_interfaces.json`。

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families db3v_strict_fewshot `
  --output-dir src\experiments\DB3V_strict_INT8_quantization_8class `
  --representative-samples 256 --batch-size 128 --num-threads 4
```
