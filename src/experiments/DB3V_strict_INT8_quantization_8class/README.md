# DB3V严格小样本链路INT8量化

本目录使用当前统一标准替代旧单种子DB3V量化实验：先在三种特征、5/10/20-shot、
四种严格策略和随机种子42/123/2026的FP32实验中，按三个seed的平均适配分数为每个
“特征×shot”选择一个策略；再量化该策略的全部三个seed，共27个TFLite模型。

每个模型均重新复测：

- 480条Xeno-canto内部验证录音；
- 未参与DB3V训练和选型的完整211条BirdSet SSW目标长录音；
- 20-shot support之外固定的10,197条DB3V共同held-out录音。

## 严格INT8定义与隔离

全部27个模型均满足：

- 只允许`TFLITE_BUILTINS_INT8`；
- 输入和输出均为`int8`；
- 图内浮点张量数为0；
- 不包含`QUANTIZE`/`DEQUANTIZE`转换算子；
- `int32` bias和累加器保留为整数卷积核的正常行为。

每个模型使用256个代表性窗口校准，样本在Xeno-canto训练特征和对应shot的DB3V
support之间等额分配。Xeno验证集、BirdSet和DB3V held-out均不参与校准；INT8结果
不反向选择特征、shot、策略、epoch或校准来源。

## FP32到INT8三seed聚合结果

每项均为三个seed的均值±样本标准差。

| 特征 | shot/策略 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---|---:|---:|---:|---:|
| MFCC | 5 / Head-Only | 47.79%±0.60% → 46.40%±0.89% | 13.49%±2.11% → 15.17%±2.32% | 8.60%±0.92% → 9.18%±0.87% | 53.94%±0.46% → 53.74%±0.47% |
| MFCC | 10 / Head-Only | 48.43%±1.35% → 47.53%±1.83% | 14.71%±2.73% → 14.08%±2.73% | 9.03%±0.89% → 8.77%±0.99% | 53.74%±2.07% → 53.09%±1.54% |
| MFCC | 20 / BN+Head+Replay | 49.64%±0.35% → 47.83%±1.88% | 17.01%±0.84% → 16.87%±1.48% | 10.53%±0.22% → 9.80%±0.40% | 56.64%±1.87% → 56.25%±1.98% |
| LogMel | 5 / Head-Only | 62.33%±0.76% → 52.21%±0.80% | 19.71%±1.22% → 24.06%±7.23% | 10.56%±0.29% → 14.98%±1.83% | 67.72%±0.12% → 64.05%±1.79% |
| **LogMel** | **10 / Head-Only** | **61.20%±0.81% → 57.54%±0.52%** | **21.37%±1.31% → 28.38%±3.82%** | **10.91%±0.22% → 15.20%±0.77%** | **67.84%±0.05% → 68.14%±0.17%** |
| LogMel | 20 / BN+Head+Replay | 60.95%±0.49% → **58.76%±0.75%** | 20.69%±0.07% → 27.12%±1.62% | 10.45%±0.06% → 14.53%±0.92% | 66.97%±0.25% → 66.79%±0.23% |
| PCEN | 5 / BN+Head+Replay | 62.58%±3.37% → 31.96%±1.37% | 16.69%±1.82% → 19.04%±4.29% | 12.67%±2.42% → 8.78%±0.33% | 67.64%±2.64% → 46.53%±0.63% |
| PCEN | 10 / Head-Only | 58.01%±2.65% → 37.09%±0.64% | 12.76%±5.45% → 22.43%±9.66% | 8.66%±0.23% → 13.37%±3.33% | 65.27%±3.39% → 52.05%±1.57% |
| PCEN | 20 / Full | **63.66%±0.44%** → 37.69%±1.61% | 11.73%±0.34% → **34.89%±2.73%** | 10.52%±0.17% → **16.04%±1.06%** | **70.50%±0.40%** → 52.52%±0.79% |

## 结论

当前DB3V严格INT8部署推荐为：

```text
Xeno-canto → LogMel → DS-CNN
→ DB3V 10-shot Head-Only → 严格INT8
```

该链路量化后DB3V Macro-F1为68.14%±0.17%，相对FP32提高0.30±0.18个百分点；
Xeno Macro-F1为57.54%±0.52%。未参与训练的BirdSet Top-1和纯单物种Macro-F1
分别为28.38%±3.82%和15.20%±0.77%。

LogMel 20-shot `BN+Head+Replay`量化后的Xeno Macro-F1更高，为58.76%±0.75%，
但DB3V Macro-F1为66.79%±0.23%，低于10-shot `Head-Only`。如果只强调Xeno保留，
可以将其作为备选。

PCEN 20-shot `Full`在FP32下取得最高DB3V Macro-F1 70.50%±0.40%，但量化后降至
52.52%±0.79%；三条PCEN链路均存在严重PTQ退化，不能作为当前INT8部署方案。

## 文件与复现

- `experiment_protocol.json`：严格定义、校准隔离和27条链路清单。
- `input_interfaces.json`：每条链路的输入形状、scale和zero-point。
- `summary.csv`：27个逐seed FP32→INT8结果。
- `aggregate.csv`：9个特征/shot/策略组合的均值与样本标准差。
- `models/<chain_id>/`：TFLite、量化元数据和三域逐粒度原始报告。
- FP32来源：`../DB3V_fewshot_ablation_multiseed_8class/aggregate.csv`及对应seed目录。
- 每个选中seed目录的`BirdSet_SSW_full_evaluation.json`保存新增FP32跨域复测。

从仓库根目录复现：

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families db3v_strict_fewshot `
  --output-dir src\experiments\DB3V_strict_INT8_quantization_8class `
  --representative-samples 256 `
  --batch-size 128 `
  --num-threads 4
```
