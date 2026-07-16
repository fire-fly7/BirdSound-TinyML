# Model_train

## 环境依赖

在项目根目录创建共用虚拟环境并安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

音频解码还需要系统中可执行的 FFmpeg。可用 `ffmpeg -version` 检查，Windows 可通过 `winget install Gyan.FFmpeg` 安装。`.venv`、下载的数据和生成的训练输出已被 Git 忽略，不会上传到仓库。

## 统一声学前端

训练、INT8 量化校准和 STM32 推理共用
`dataset_processing/acoustic_frontend_config.json`。每个数据集、模型和转换结果都会记录该配置的 SHA-256；哈希或输入形状不一致时，脚本会直接报错，避免训练端和单片机端采用不同的 MFCC 参数。

当前模型输入配置：

- 16 kHz 单声道 PCM16，固定 1 秒（16,000 个采样点）
- 1,024 点 FFT，步长 512，不进行居中填充
- 周期 Hann 窗
- 40 个 Slaney 刻度、Slaney 面积归一化的 Mel 滤波器，范围 0–8 kHz
- 功率谱转 dB：`amin=1e-10`、参考值 1.0、`top_db=80`
- 正交归一化 DCT-II，保留第 0–12 个系数
- MFCC 输出为 `30 x 13`，模型输入为 `30 x 13 x 1`

修改 JSON 后，重新生成 STM32 常量和兼容用 Mel 滤波器头文件：

```powershell
python dataset_processing/generate_frontend_assets.py
python dataset_processing/generate_frontend_assets.py --check
```

## 生成 MFCC 数据

输入目录按类别建立子目录，每个类别目录中放置 WAV 文件：

```text
audio_root/
  class_a/*.wav
  class_b/*.wav
```

自动按原始音频文件划分训练、验证和测试集：

```powershell
python dataset_processing/data_sugment_MFCC.py `
  --input-dir row_dataset/birdset_HSN/train `
  --output-dir dataset_processing/output/MFCC_dataset
```

脚本输出 `.npy` 数据、标签映射、切片清单和 `feature_config.json`。同一条原始录音的相邻切片不会跨数据集，尾部不足 1 秒时默认补零；如需丢弃尾部，可增加 `--drop-remainder`。

## 训练和完整 INT8 转换

训练时会校验 `feature_config.json`，并在 `.h5` 模型旁生成同名 `.frontend.json`：

```powershell
python "src/model_train&test/train.py" `
  --data-dir dataset_processing/output/MFCC_dataset `
  --model DS_CNN_Model
```

INT8 转换会同时校验当前 JSON、代表性数据集和模型侧车文件：

```powershell
python "src/model_train&test/convert/TFlite_converter.py" `
  --model "src/model_train&test/TinyML_model/DS_CNN_Model.h5" `
  --calibration-data dataset_processing/output/MFCC_dataset/train_data.npy
```

## STM32 CMSIS-DSP

实现位于 `dataset_processing/stm32_frontend`。将目录中的 `.c` 和 `.h` 文件加入 STM32 工程，启用 CMSIS-DSP TransformFunctions，先调用 `MfccFrontend_Init`，再向 `MfccFrontend_Compute` 传入固定 16,000 个 PCM16 采样点。输出按帧优先排列，共 `30 x 13` 个 `float32_t`。

生成确定性的 PCM 和 Python MFCC 参考文件：

```powershell
python dataset_processing/verify_frontend_consistency.py
```

参考文件写入 `dataset_processing/output/frontend_reference`。让 STM32 处理其中的 `reference_pcm.csv`，将结果导出为包含 390 个值的 `30 x 13` CSV，然后比较：

```powershell
python dataset_processing/verify_frontend_consistency.py `
  --mcu-mfcc stm32_mfcc.csv
```
