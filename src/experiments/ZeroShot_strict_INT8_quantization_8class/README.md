# Xeno-canto零样本严格INT8基准

本目录只保存当前仍有效的三条Xeno-canto零样本严格INT8链路。FP32来源为
`../Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.h5`，模型选择、
量化校准和三域评估均不使用BirdSet或DB3V support。

三个模型均满足：

- 只允许`TFLITE_BUILTINS_INT8`；
- 输入和输出均为`int8`；
- 图内浮点张量数为0；
- 不包含`QUANTIZE`或`DEQUANTIZE`算子；
- 每条链路仅使用对应Xeno-canto训练特征中的256个代表性窗口校准。

BirdSet固定使用公共20-shot held-out：197条长录音、18,265个五秒片段、
91,325个一秒窗口；DB3V使用完整10,658条单标签录音。

| 特征 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC | 49.62% → 48.77% | 16.99% → 15.36% | 9.74% → 9.40% | 50.47% → 51.25% |
| LogMel | **63.56% → 59.01%** | 17.78% → 18.42% | **9.88% → 10.22%** | **67.62% → 65.94%** |
| PCEN | 52.69% → 35.07% | **19.52% → 21.40%** | 8.04% → 9.20% | 59.93% → 50.24% |

LogMel量化后仍是当前通用零样本INT8推荐。PCEN的BirdSet Top-1较高，但在Xeno和
DB3V上存在明显PTQ退化。

目录内容：

- `models/<chain_id>/`：原始TFLite、量化元数据和三域评估报告；
- `summary.csv`：三条FP32→INT8结果；
- `experiment_protocol.json`：数据隔离、校准和公共测试协议；
- `input_interfaces.json`：输入形状、scale和zero-point；
- `artifact_hashes.csv`：迁移后九个模型工件的SHA-256。

模型工件由原混合目录迁入时保持字节不变；原始评估JSON内部若包含旧输出路径，
以`artifact_hashes.csv`和当前目录结构为准，指标值未改写。

从仓库根目录复现：

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families zero_shot `
  --output-dir src\experiments\ZeroShot_strict_INT8_quantization_8class `
  --representative-samples 256 `
  --batch-size 128 `
  --num-threads 4
```
