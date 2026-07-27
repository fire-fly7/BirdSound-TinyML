# BirdSet严格小样本链路INT8量化

本目录对BirdSet grouped 5/10/20-shot严格实验中按三个seed平均适配分数选出的
9组FP32策略进行后训练静态量化。每组保留42、123、2026三个seed，共27个TFLite
模型；每个模型均重新复测Xeno-canto验证集、共同BirdSet 20-shot held-out和
完整DB3V。

## 严格INT8定义与隔离

全部27个模型均满足：

- 只允许`TFLITE_BUILTINS_INT8`；
- 输入和输出均为`int8`；
- 图内浮点张量数为0；
- 不包含`QUANTIZE`/`DEQUANTIZE`转换算子；
- `int32` bias和累加器保留为整数卷积核的正常行为。

每个模型使用256个代表性窗口校准，样本在Xeno-canto训练特征和匹配shot的
BirdSet support之间等额分配。BirdSet held-out、Xeno验证集和DB3V均不参与校准，
INT8结果也不反向选择特征、shot、策略、epoch、阈值或校准来源。

## FP32到INT8三seed聚合结果

每项均为三个seed的均值；完整均值、样本标准差、绝对变化和保留率见
`aggregate.csv`。

| 特征 | shot/策略 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---|---:|---:|---:|---:|
| MFCC | 5 / BN+Head+Replay | 48.41% → 48.39% | 20.06% → 20.15% | 13.32% → 14.45% | 48.45% → 48.26% |
| MFCC | 10 / BN+Head+Replay | 47.89% → 47.81% | 19.88% → 20.69% | 13.44% → 15.20% | 48.45% → 48.07% |
| MFCC | 20 / BN+Head+Replay | 48.53% → 47.72% | 23.89% → **24.12%** | 17.57% → **18.40%** | 49.23% → 48.70% |
| LogMel | 5 / BN+Head+Replay | 56.10% → 53.14% | 20.31% → 20.22% | 13.32% → 13.20% | 59.47% → 57.85% |
| LogMel | 10 / BN+Head+Replay | 56.11% → 54.40% | 20.39% → 23.09% | 13.66% → 14.19% | 59.33% → 57.84% |
| LogMel | 20 / Head-Only | 57.81% → **55.62%** | 18.66% → 21.18% | 10.62% → 12.21% | 63.73% → **63.32%** |
| PCEN | 5 / BN+Head | 55.27% → 17.97% | 36.11% → 19.78% | 27.11% → 4.25% | 61.41% → 27.80% |
| PCEN | 10 / BN+Head | 54.76% → 19.03% | 37.56% → 19.84% | 28.06% → 4.34% | 60.56% → 27.62% |
| PCEN | 20 / BN+Head | 57.85% → 16.85% | 29.68% → 19.28% | 23.42% → 4.18% | 63.67% → 28.53% |

MFCC三条链路的量化变化最小，且MFCC 20-shot在INT8下取得最高BirdSet Top-1和
纯单物种Macro-F1。若部署目标需要兼顾Xeno基准保留、BirdSet声景适配和DB3V跨域
绝对精度，当前较均衡的严格INT8链路是
`Xeno-canto → LogMel → DS-CNN → BirdSet 20-shot Head-Only → INT8`：
Xeno和DB3V Macro-F1分别为55.62%和63.32%，BirdSet Top-1为21.18%。

PCEN在FP32下的BirdSet结果最高，但三个shot、全部seed量化后均发生严重退化，
Xeno Macro-F1下降35.73–40.99个百分点，DB3V Macro-F1下降32.94–35.14个百分点。
因此PCEN不适合直接使用当前PTQ配置；部署前需要量化感知训练并重新执行全部
held-out评估。

## 文件与复现

- `experiment_protocol.json`：严格定义、校准隔离和27条链路清单。
- `input_interfaces.json`：每条链路的输入形状、scale和zero-point。
- `summary.csv`：27个逐seed FP32→INT8结果。
- `aggregate.csv`：9个特征/shot/策略组合的均值与样本标准差。
- `models/<chain_id>/`：TFLite、量化元数据和三域逐粒度原始报告。
- FP32来源：`../BirdSet_fewshot_ablation_multiseed_8class/aggregate.csv`及对应
  seed目录。
- 数据官方来源与许可见FP32实验目录README和项目根README。

从仓库根目录复现：

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families birdset_strict_fewshot `
  --output-dir src\experiments\BirdSet_strict_INT8_quantization_8class
```
