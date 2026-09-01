# Xeno-canto零样本严格INT8基准

本目录保存MFCC、LogMel、PCEN三条Xeno-canto零样本DS-CNN的FP32–INT8同模型、同测试集配对。模型选择、量化校准和评估均不使用BirdSet或DB3V support。

三个模型均只允许`TFLITE_BUILTINS_INT8`，输入/输出均为int8，浮点张量数为0，无`QUANTIZE`/`DEQUANTIZE`算子；每条链路只使用对应Xeno训练集的256个代表性窗口校准。

BirdSet固定使用197条长录音、18,265个五秒片段的公共test。DB3V主指标在完整数据的1,363条独立原始来源录音上计算；10,658个八秒块与85,264个一秒切片保留为次级粒度。

| 特征 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V源录音Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC | 49.62% → 48.77% | 16.99% → 15.36% | 9.74% → 9.40% | 55.55% → 56.75% |
| **LogMel** | **63.56% → 59.01%** | 17.78% → 18.42% | **9.88% → 10.22%** | **72.87% → 70.05%** |
| PCEN | 52.69% → 35.07% | **19.52% → 21.40%** | 8.04% → 9.20% | 66.15% → 53.48% |

LogMel量化后仍是通用零样本推荐。PCEN虽在BirdSet Top-1上最高，但Xeno和DB3V均发生明显PTQ退化。

- `summary.csv`：三条FP32–INT8配对。
- `experiment_protocol.json`：隔离、校准、公共测试和DB3V源录音级主指标。
- `input_interfaces.json`：输入形状及每模型scale/zero-point。
- `models/<chain_id>/evaluation.json`：三域逐粒度原始报告。
- `artifact_hashes.csv`：模型工件SHA-256。

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families zero_shot `
  --output-dir src\experiments\ZeroShot_strict_INT8_quantization_8class `
  --representative-samples 256 --batch-size 128 --num-threads 4
```
