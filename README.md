# BirdSound-TinyML · 鸟声分类训练与部署

面向八类鸟声的 TinyML 实验项目：以 Xeno-canto 训练 DS-CNN，在 DB3V 与 BirdSet SSW 上进行独立评估及小样本适配，并导出严格 INT8 TFLite 模型。

配套固件：[BirdSound-STM32](https://github.com/fire-fly7/BirdSound-STM32)。两个仓库以 `main` 为当前开发入口，训练端的 `shared/deployment_contract.json` 是共享配置的权威来源。

## 当前实现

- 三种音频特征：MFCC、LogMel、PCEN；单声道 16 kHz，1 秒窗口。
- MFCC 输入为 `[1,32,13,1]`，LogMel/PCEN 为 `[1,32,40,1]`，输出为八类。
- 外部测试与 support 隔离，支持 5/10/20-shot、多随机种子适配与统一 INT8 复测。
- 统一生成 Python 配置、标签、固件前端 C 表、模型数组、部署清单与哈希。
- C/Python 共用 float32、nearest-even、先舍入后加零点的量化约定。

## 目录

| 路径 | 内容 |
|---|---|
| `shared/` | 共享配置、校验和量化接口 |
| `tools/` | 前端表、模型部署和统一生成工具 |
| `src/row_dataset/` | 数据获取脚本；原始音频不进入 Git |
| `src/dataset_processing/` | 三特征预处理、标签和数据划分 |
| `src/experiments/` | 训练、独立评估、小样本适配和结果 |
| `docs/RESEARCH_GUIDE.md` | 数据角色、完整实验流程及历史结果 |

## 快速开始

运行环境需要 TensorFlow/Keras、NumPy、librosa、soundfile 等；音频获取及标准化还需 FFmpeg。历史严格 INT8 对拍使用 TensorFlow 2.19.0。完整训练命令与依赖使用方式见[研究指南](docs/RESEARCH_GUIDE.md)和[实验目录说明](src/experiments/README.md)。数据需自行获取；仓库中的模型与报告不意味着原始音频也已包含。

从现有 LogMel INT8 模型生成部署包（在仓库根目录执行）：

```sh
python tools/generate_deployment.py \
  --model src/experiments/ZeroShot_strict_INT8_quantization_8class/models/zero_shot_logmel/DS_CNN_Model.int8.tflite \
  --feature LOGMEL \
  --source-commit "$(git rev-parse HEAD)" \
  --output-dir /tmp/deployment/zero_shot_logmel
```

两个仓库位于同一文件系统时，可增加 `--firmware-root /path/to/BirdSound-STM32` 同步共享配置和生成器。校验同步状态：

```sh
python tools/generate_deployment.py \
  --firmware-root /path/to/BirdSound-STM32 \
  --output-dir /tmp/deployment/check --check
```

Windows 可将 `python` 替换为现有 `.venv\Scripts\python.exe`，并将输出目录改为本机路径。远程固件需传输部署包，不能直接把 SSH 地址作为 `--firmware-root`。

## 验证与结果边界

2026-10-06 完成共享生成改造验证：三前端在五种测试音频上与改造前 Python 输出完全一致，C 表逐字节一致；50,820 个 C/Python 量化用例通过；57 个现有模型通过部署生成器兼容性检查。配置哈希、模型哈希和标签顺序不符会被拒绝。兼容性检查不能替代准确率评估或实际推理验证。

已有 216 个 FP32 适配模型和 57 个严格 INT8 模型（3 个零样本、27 个 DB3V、27 个 BirdSet）。正式指标应引用具体实验目录的原始报告，并说明划分、随机种子与数据角色；历史结果不会因更新生成器自动成为新版结果。

新版固件已编译，尚未完成新版实板对拍。实时麦克风集成、功耗测量及新版全量指标复测仍待完成。

## 版本与数据

当前仅维护 `main`；提交历史保留，用于定位实验和追溯变化。部署清单记录源码提交、配置和模型哈希，发布入口不再指向旧固件版本。

原始数据、生成特征、缓存、日志和板端测试语料按 `.gitignore` 排除。数据集许可与论文引用见[研究指南](docs/RESEARCH_GUIDE.md)。勿提交 API 密钥；Xeno-canto 密钥通过 `XENO_CANTO_API_KEY` 环境变量提供。
