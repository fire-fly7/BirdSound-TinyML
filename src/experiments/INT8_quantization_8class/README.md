# 严格INT8量化：零样本与旧单种子归档

本目录保存15条已经由FP32实验选中的模型链路的严格INT8模型与精度报告：

- 3条 Xeno-canto 零样本基准：MFCC、LogMel、PCEN；
- 9条 DB3V 5/10/20-shot 选中链路：每种特征在每个shot下的FP32选中策略；
- 3条 BirdSet grouped 5-shot 选中链路。

其中3条零样本结果继续作为当前基准；12条小样本结果来自旧单种子、
`head/last_block/all`策略实验，仅用于历史追溯，不再参与当前推荐。当前严格
多种子小样本结果分别位于：

- `../DB3V_strict_INT8_quantization_8class/`
- `../BirdSet_strict_INT8_quantization_8class/`

量化结果不参与重新选择特征、shot、微调策略或轮次。零样本模型只使用Xeno-canto
训练特征校准；小样本模型只额外加入对应support。BirdSet、DB3V held-out和
Xeno-canto验证集均不用于校准。

## 严格INT8条件

- `TFLITE_BUILTINS_INT8` 是唯一允许的TFLite算子集；
- 输入和输出均为 `int8`；
- 图内浮点张量数为0；
- 图内没有 `QUANTIZE` 或 `DEQUANTIZE` 算子；
- `int32` 偏置和乘加累积属于标准整数内核，不视为混合浮点；
- 15个模型均通过上述检查，大小为49,968–49,984 bytes。

## 输入接口

| 特征 | 浮点特征输入 | TFLite输入 | 量化公式 |
|---|---|---|---|
| MFCC | `float32 [batch,32,13]` | `int8 [batch,32,13,1]` | `clip(round(x/scale)+zero_point,-128,127)` |
| LogMel | `float32 [batch,32,40]` | `int8 [batch,32,40,1]` | 同上 |
| PCEN | `float32 [batch,32,40]` | `int8 [batch,32,40,1]` | 同上 |

LogMel和PCEN虽然形状相同，但语义和校准数据不同，接口必须显式指定特征名，不能
只根据形状互换。每条模型的实际scale和zero-point见 `input_interfaces.json`。

## 零样本结果

| 特征 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC | 49.62% → 48.77% (-0.85 pp) | 17.01% → 15.43% (-1.57 pp) | 9.72% → 9.41% (-0.31 pp) | 50.47% → 51.25% (+0.78 pp) |
| LogMel | 63.56% → 59.01% (-4.55 pp) | 17.87% → 18.48% (+0.61 pp) | 9.95% → 10.32% (+0.37 pp) | 67.62% → 65.94% (-1.69 pp) |
| PCEN | 52.69% → 35.07% (-17.62 pp) | 19.37% → 21.55% (+2.17 pp) | 8.04% → 9.28% (+1.24 pp) | 59.93% → 50.24% (-9.69 pp) |

MFCC量化最稳定；LogMel量化损失更大，但INT8后的绝对Xeno和DB3V Macro-F1仍是
三条零样本链路中最高；PCEN不适合直接做训练后静态INT8量化。

## DB3V旧单种子小样本结果（历史归档）

DB3V统一使用20-shot support之外的10,197条共同held-out；BirdSet使用完整目标
子集，因为DB3V适配没有读取BirdSet。

| 链路 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC/5-shot/head | 46.10% → 46.05% (-0.05 pp) | 10.72% → 12.60% (+1.88 pp) | 7.00% → 7.91% (+0.91 pp) | 55.58% → 55.55% (-0.04 pp) |
| MFCC/10-shot/head | 47.87% → 47.47% (-0.40 pp) | 11.93% → 11.74% (-0.19 pp) | 7.48% → 7.39% (-0.08 pp) | 56.94% → 56.46% (-0.48 pp) |
| MFCC/20-shot/head | 40.47% → 40.47% (0.00 pp) | 24.17% → 23.92% (-0.26 pp) | 14.92% → 15.08% (+0.16 pp) | 61.38% → 60.81% (-0.58 pp) |
| LogMel/5-shot/head | 60.05% → 53.45% (-6.60 pp) | 18.81% → 30.86% (+12.05 pp) | 9.19% → 15.31% (+6.12 pp) | 67.64% → 64.68% (-2.96 pp) |
| **LogMel/10-shot/last_block** | **62.13% → 58.54% (-3.58 pp)** | **18.86% → 22.85% (+3.99 pp)** | **9.90% → 12.11% (+2.21 pp)** | **68.59% → 69.70% (+1.12 pp)** |
| LogMel/20-shot/all | 63.45% → 58.01% (-5.44 pp) | 19.47% → 20.25% (+0.77 pp) | 10.46% → 11.45% (+0.99 pp) | 68.86% → 67.91% (-0.96 pp) |
| PCEN/5-shot/all | 59.92% → 33.26% (-26.66 pp) | 9.81% → 25.72% (+15.91 pp) | 9.42% → 14.24% (+4.83 pp) | 65.82% → 47.13% (-18.69 pp) |
| PCEN/10-shot/head | 60.74% → 38.55% (-22.20 pp) | 12.70% → 11.77% (-0.93 pp) | 9.86% → 9.62% (-0.23 pp) | 65.95% → 52.00% (-13.95 pp) |
| PCEN/20-shot/last_block | 61.51% → 35.07% (-26.43 pp) | 9.70% → 37.85% (+28.15 pp) | 9.16% → 17.67% (+8.51 pp) | 70.60% → 54.46% (-16.14 pp) |

按当时旧单种子结果，INT8部署的DB3V推荐链路曾为
`Xeno-canto → LogMel → DS-CNN → DB3V 10-shot last_block`。它在共同held-out上
达到69.70% Macro-F1，同时保持58.54%的Xeno Macro-F1；FP32最高的PCEN 20-shot
在严格INT8下退化到54.46%。该推荐已由严格三种子
`LogMel 10-shot head_only`结果替代。

## BirdSet旧单种子小样本结果（历史归档）

BirdSet使用与support按长录音隔离的201条held-out；DB3V完整集没有参与本轮训练。

| 链路 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC/5-shot/all | 42.43% → 39.62% (-2.81 pp) | 24.23% → 24.97% (+0.74 pp) | 17.75% → 18.72% (+0.97 pp) | 41.58% → 41.81% (+0.23 pp) |
| **LogMel/5-shot/head** | **55.80% → 53.68% (-2.12 pp)** | **21.09% → 23.82% (+2.73 pp)** | **15.15% → 16.97% (+1.82 pp)** | **57.94% → 58.08% (+0.14 pp)** |
| PCEN/5-shot/head | 59.44% → 34.35% (-25.09 pp) | 35.70% → 37.37% (+1.67 pp) | 24.66% → 17.47% (-7.19 pp) | 66.66% → 45.97% (-20.69 pp) |

如果只看BirdSet Top-1，PCEN INT8仍为37.37%；但其纯单物种Macro-F1下降7.19个
百分点，Xeno和DB3V分别下降25.09和20.69个百分点，因此不能视为稳定提升。兼顾
多标签声景、纯单物种和跨域保留时，严格INT8推荐
`Xeno-canto → LogMel → DS-CNN → BirdSet grouped 5-shot head`。该历史推荐已由
严格5/10/20-shot三种子结果替代。

## 历史结论与限制

- MFCC是当前最耐PTQ量化的特征，所有小样本链路的主要Macro-F1变化均在约3个百分点内。
- LogMel是绝对精度与量化稳定性的较优折中，但不同微调轮次仍会造成2–7个百分点的Xeno损失。
- PCEN的输入饱和率小于0.001%，但Xeno/DB3V Macro-F1普遍下降约10–27个百分点，
  说明输入接口截断不是主因，更可能是内部激活量化敏感，仍需逐层误差分析确认。
- BirdSet Top-1可能因预测偏向常见多标签类别而上升，必须同时检查纯单物种
  Macro-F1、Xeno和DB3V，不能单独把Top-1上升解释为量化增益。
- 当前是训练后静态量化；若必须部署PCEN，应单独进行量化感知训练并重新执行全部
  held-out评估，不能使用本次PCEN PTQ模型。

## 文件

- `summary.csv`：15条链路的统一FP32→INT8指标、差值、保留率和输入饱和率；
- `input_interfaces.json`：每条链路的输入形状、scale和zero-point；
- `experiment_protocol.json`：链路清单、严格INT8定义和校准隔离规则；
- `models/<chain_id>/DS_CNN_Model.int8.tflite`：严格INT8模型；
- `models/<chain_id>/DS_CNN_Model.int8_metadata.json`：张量类型、算子、大小和校准来源；
- `models/<chain_id>/evaluation.json`：逐类混淆矩阵、切片/片段/录音粒度指标。

复现实验：

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families zero_shot db3v_fewshot birdset_fewshot `
  --output-dir src\experiments\INT8_quantization_8class `
  --representative-samples 256 `
  --batch-size 128 `
  --num-threads 4
```
