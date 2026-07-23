# 鸟类鸣声分类项目

本项目使用 Xeno-canto 鸟鸣录音训练分类器，并将 DB3V 保持为完全独立的区域测试集。当前推荐实验是 **8 类、每类 300 条 Xeno-canto 原始录音、A–E 评分等级** 的 DS-CNN 基线；DB3V 不参与训练、验证或模型选择。

## 当前八类配置

| ID | 类别 |
|---:|---|
| 0 | `Agelaius_phoeniceus` |
| 1 | `Cardinalis_cardinalis` |
| 2 | `Certhia_americana` |
| 3 | `Corvus_brachyrhynchos` |
| 4 | `Setophaga_aestiva` |
| 5 | `Setophaga_ruticilla` |
| 6 | `Spinus_tristis` |
| 7 | `Turdus_migratorius` |

当前原始集位于 `row_dataset/row_bird_dataset_A_next/`，共 2,400 条录音。每个文件均为 16 kHz、单声道、16-bit PCM WAV；全局 ID 与音频哈希已去重，并与 DB3V 来源 ID 隔离。

## 项目结构

```text
.
├── README.md
├── row_dataset/
│   ├── data_crawler_xeno_canto.py          # Xeno-canto 爬虫；支持八类、A–E、DB3V 来源排除
│   ├── data_crawler_DB3V.py                # DB3V 原始数据下载/整理
│   ├── DB3V/
│   │   └── extracted/data_wav_8s_2/         # 原始 DB3V WAV；仅用于独立评估/后续小样本实验
│   ├── BirdSet_SSW/                          # BirdSet SSW 元数据缓存；原始分片不落盘
│   ├── row_bird_dataset_A_next/            # 当前八类 Xeno-canto 原始训练数据
│   └── xeno_removed_db3v_overlap.json      # 历史 DB3V/Xeno-canto 来源重叠记录
├── dataset_processing/
│   ├── data_sugment_MFCC.py                # Xeno-canto 八类 MFCC 预处理与无泄漏切分
│   ├── data_sugment_MFCC_DB3V.py           # DB3V 八类 MFCC 独立测试集预处理
│   ├── data_prepare_birdset_ssw.py          # BirdSet SSW 流式筛选及多标签 MFCC 预处理
│   ├── data_sugment_logMel.py              # 备选 Log-Mel 特征实验
│   ├── data_sugment_PCEN.py                # 备选 PCEN 特征实验
│   ├── label_map_8class.json               # 当前八类连续标签映射
│   └── output/
│       ├── MFCC_dataset_A_8class/          # 当前 Xeno-canto 训练/验证特征
│       ├── MFCC_dataset_DB3V_8class/       # 当前 DB3V 独立测试特征
│       └── MFCC_dataset_BirdSet_SSW_8class/ # 当前 BirdSet 多标签声景测试特征
└── src/model_train&test/
    ├── Tranin.py                           # 训练入口；按录音级验证 Macro-F1 选择最佳权重
    ├── evaluate_db3v.py                    # DB3V 独立区域评估，输出切片级和录音级指标
    ├── evaluate_birdset_ssw.py             # BirdSet 多标签切片级和片段级评估
    ├── convert.py                           # 可选模型转换入口；不由训练脚本自动调用
    ├── construct_model/
    │   ├── DS_CNN_Model.py
    │   ├── CNN_Model.py
    │   ├── BC_ResNet.py
    │   └── MobileNetV2.py
    └── TinyML_model_8class/                # 当前八类模型、标签侧车和 DB3V 报告
```

目录树只列出当前八类流程的规范路径。并行抓取暂存目录、旧实验工件、`__pycache__/`、`.venv/` 和日志文件都不是当前训练输入或源码结构。

## 推荐工作流

使用项目虚拟环境执行以下命令。重新抓取时不要使用 `--allow-partial`，以确保每个类别满足 300 条有效录音的下限。

```powershell
# 1. 获取八类 Xeno-canto 原始数据（默认接受 A,B,C,D,E 评分）
& .\.venv\Scripts\python.exe row_dataset\data_crawler_xeno_canto.py `
  --eligible-eight `
  --target-per-species 300 `
  --min-per-species 300 `
  --output-dir row_dataset\row_bird_dataset_A_next

# 2. 生成 Xeno-canto 训练/验证 MFCC 特征
& .\.venv\Scripts\python.exe dataset_processing\data_sugment_MFCC.py `
  --input-dir row_dataset\row_bird_dataset_A_next `
  --output-dir dataset_processing\output\MFCC_dataset_A_8class

# 3. 生成同标签顺序的 DB3V 独立测试特征
& .\.venv\Scripts\python.exe dataset_processing\data_sugment_MFCC_DB3V.py `
  --output-dir dataset_processing\output\MFCC_dataset_DB3V_8class `
  --xeno-label-map dataset_processing\output\MFCC_dataset_A_8class\label_map.json

# 4. 训练 DS-CNN；EarlyStopping 监控录音级 Macro-F1
$env:TF_CPP_MIN_LOG_LEVEL = '2'
& .\.venv\Scripts\python.exe "src\model_train&test\Tranin.py" `
  --model DS_CNN_Model `
  --dataset-dir dataset_processing\output\MFCC_dataset_A_8class `
  --model-dir "src\model_train&test\TinyML_model_8class" `
  --epochs 30 --patience 8

# 5. 在完全独立的 DB3V 上测试
& .\.venv\Scripts\python.exe "src\model_train&test\evaluate_db3v.py" `
  --models DS_CNN_Model `
  --dataset-dir dataset_processing\output\MFCC_dataset_DB3V_8class `
  --model-dir "src\model_train&test\TinyML_model_8class"
```

## 数据与评估约定

- Xeno-canto 按完整录音的 `session_key` 切分：当前为每类 240 条训练录音、60 条验证录音，训练/验证之间没有共享录音或会话。
- 每条录音最多贡献 8 个均匀分布的 1 秒片段；训练权重使每条录音总权重相同，避免长录音主导优化。
- 当前 MFCC 特征形状为 `(32, 13)`：Xeno-canto 训练/验证数组分别为 `(14545, 32, 13)` 和 `(3652, 32, 13)`；DB3V 数组为 `(85264, 32, 13)`，对应 10,658 条录音。
- DB3V 保持为独立地区测试集。评估时会同时报告 1 秒切片指标，以及将一条 DB3V 录音的 8 个 softmax 向量平均后的录音级指标。
- 模型、`*.labels.json` 标签侧车、训练历史、验证摘要与 DB3V JSON/CSV 报告会保存在同一个模型目录中。转换为 TFLite 或 C 头文件时，单独运行 `convert.py`。
- Xeno-canto API 密钥不应写入 README；如需覆盖本地配置，可使用 `XENO_CANTO_API_KEY` 环境变量。音频标准化依赖 FFmpeg。

## 数据来源、许可与实验角色

| 数据 | 上游来源 | 本实验中的角色 | 许可/引用说明 |
|---|---|---|---|
| Xeno-canto | [Xeno-canto](https://xeno-canto.org/) 及其 [API](https://xeno-canto.org/explore/api) | 2,400 条焦点录音用于训练和内部验证 | 每条录音的 Creative Commons 许可保存在对应 `metadata.jsonl`；再分发或商用时必须逐条遵守 |
| DB3V | [Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734) | 完全独立的三地区单标签测试，不参与训练、早停或模型选择 | Zenodo记录标注为 CC BY 4.0；使用时应引用数据集记录 |
| BirdSet | [官方数据集](https://huggingface.co/datasets/DBD-research-group/BirdSet)、[官方代码](https://github.com/DBD-research-group/BirdSet)、[ICLR 2025 论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/484d254ff80e99d543159440a06db0de-Abstract-Conference.html) | SSW `test_5s` 作为完全独立的真实多标签声景测试 | BirdSet代码为 BSD-3-Clause；底层音频仍遵循各来源许可 |
| SSW底层声景 | [Zenodo 7079380](https://zenodo.org/records/7079380) | BirdSet SSW评估音频的原始来源 | 当前保留片段的清单均标记为 CC BY 4.0，逐片段来源和许可保存在 `manifest.json` |

BirdSet论文引用：

```bibtex
@inproceedings{rauch2025birdset,
  title     = {BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics},
  author    = {Lukas Rauch and Raphael Schwinger and Moritz Wirth and
               Ren{\'e} Heinrich and Denis Huseljic and Marek Herde and
               Jonas Lange and Stefan Kahl and Bernhard Sick and
               Sven Tomforde and Christoph Scholz},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```

## 当前实验进度

截至 **2026-07-24**，已经完成：

1. 确定八个目标物种和固定标签顺序。
2. 获取并清洗每类 300 条、总计 2,400 条 Xeno-canto 录音；执行来源 ID、音频哈希和 DB3V 来源隔离。
3. 按完整录音会话切分为每类 240 条训练录音和 60 条验证录音；训练与验证不共享录音或会话。
4. 完成统一的 16 kHz、单声道、一秒窗口、`(32, 13)` MFCC 处理。
5. 训练八分类 DS-CNN，以录音级 Macro-F1 监控 EarlyStopping 并保存最佳权重。
6. 完成 DB3V 三个地区的 1 秒切片级及 8 秒录音级独立评估。
7. 流式读取 BirdSet SSW 全部四个 `test_5s` 分片（约 7.85 GB），未在本地保存完整原始分片。
8. 从 205,200 个 SSW 片段中筛出 19,678 个命中目标物种的五秒多标签片段，生成 98,390 个一秒 MFCC 窗口。
9. 完成 BirdSet 的一秒切片级、五秒片段级以及纯单物种片段级评估。

尚未完成的实验包括：BirdSet训练子集增量训练、背景/未知类别、多鸟混合增强、量化后精度评估以及其他模型在当前八类规范数据上的公平复测。

## 当前 DS-CNN 基准模型

当前基准模型为 `DS_CNN_Model`：

- 输入：一秒音频对应的 `(32, 13, 1)` MFCC；
- 输出：八类 softmax；
- 训练数据：Xeno-canto；
- 独立测试：DB3V 和 BirdSet SSW；
- 模型文件：`src/model_train&test/TinyML_model_8class/DS_CNN_Model.h5`。

内部留出验证以完整录音为评估单位：

| 数据与粒度 | 数量 | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Xeno-canto内部验证，录音级 | 480条录音 | 50.00% | 50.00% | **49.62%** |

结果来源：`src/model_train&test/TinyML_model_8class/DS_CNN_Model.validation.json`。

## DB3V 独立区域测试结果

DB3V 每条八秒录音产生八个一秒 MFCC 窗口。“切片级”独立评价每个窗口；“录音级”先平均同一录音的八个 softmax 向量再分类。

### DB3V汇总

| 粒度 | 数量 | Accuracy | Balanced Accuracy | Macro-F1 | Top-3 |
|---|---:|---:|---:|---:|---:|
| 一秒切片级 | 85,264个切片 | 39.80% | 37.34% | 35.48% | 67.69% |
| 八秒录音级 | 10,658条录音 | **56.97%** | **52.14%** | **50.47%** | **84.79%** |

### DB3V分地区录音级结果

| 地区 | 录音数 | Accuracy | Balanced Accuracy | Macro-F1 | Top-3 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,251 | 56.54% | 50.09% | 48.19% | 85.36% |
| 2 | 752 | **71.28%** | **64.05%** | **57.38%** | **89.23%** |
| 3 | 4,655 | 55.15% | 50.53% | 47.99% | 83.44% |

地区2表现最高但样本量最少。时间聚合相对切片级使汇总 Accuracy 提高约 17.2 个百分点、Macro-F1 提高约 15.0 个百分点。

结果来源：

- `src/model_train&test/TinyML_model_8class/DB3V_evaluation.json`
- `src/model_train&test/TinyML_model_8class/DB3V_evaluation_summary.csv`

## BirdSet SSW 独立声景评估

BirdSet 仅作为外部数据源使用，不需要迁移到 PyTorch。处理脚本流式读取
SSW 的四个 `test_5s` 分片，将命中当前八类的 32 kHz 多标签声景转换为
16 kHz、五个一秒窗口及 `(32, 13)` MFCC；完整原始音频分片不会落盘。

```powershell
# 下载元数据并检查物种覆盖，不读取约 7.85 GB 音频分片
& .\.venv\Scripts\python.exe dataset_processing\data_prepare_birdset_ssw.py `
  --metadata-only

# 流式生成 BirdSet SSW 多标签 MFCC 测试集
& .\.venv\Scripts\python.exe dataset_processing\data_prepare_birdset_ssw.py

# 用当前八类 DS-CNN 进行多标签声景评估
& .\.venv\Scripts\python.exe "src\model_train&test\evaluate_birdset_ssw.py" `
  --models DS_CNN_Model
```

输出位于 `dataset_processing/output/MFCC_dataset_BirdSet_SSW_8class/`，报告位于
`src/model_train&test/TinyML_model_8class/BirdSet_SSW_evaluation.json`。SSW 中
`Setophaga_ruticilla` 只有两个多标签正例且没有纯单物种片段，因此报告分别给出
“预测命中任一目标标签”的多标签指标，以及只覆盖其余七类的纯单物种指标。

### BirdSet评估粒度与结果

多标签 Top-1/Top-3 表示概率最高的一个/三个类别是否命中该样本的任意真实目标标签，不能与 DB3V 单标签 Accuracy 直接等价比较。

| 粒度 | 数量 | Top-1任意目标命中率 | Top-3任意目标命中率 | 单标签Macro-F1 |
|---|---:|---:|---:|---:|
| 一秒多标签切片级 | 98,390个切片 | 15.62% | 41.17% | 不适用 |
| 五秒多标签片段级 | 19,678个片段 | **17.32%** | **44.33%** | 不适用 |
| 五秒纯单物种片段级 | 10,230个片段 | 16.64% Accuracy | 未统计 | 9.77%（仅7个有支持类别） |

### BirdSet五秒片段级分类召回

| 物种 | 多标签正例片段 | Top-1召回率 |
|---|---:|---:|
| `Agelaius_phoeniceus` | 6,480 | 5.82% |
| `Cardinalis_cardinalis` | 3,781 | 2.72% |
| `Certhia_americana` | 196 | **62.76%** |
| `Corvus_brachyrhynchos` | 3,861 | **65.84%** |
| `Setophaga_aestiva` | 353 | 2.27% |
| `Setophaga_ruticilla` | 2 | 0.00% |
| `Spinus_tristis` | 3,030 | 0.53% |
| `Turdus_migratorius` | 3,189 | 7.53% |

模型在 BirdSet 上的预测明显集中于 `Certhia_americana` 和 `Corvus_brachyrhynchos`，说明从 Xeno-canto 焦点录音迁移到远距离、多鸟及背景噪声声景时存在明显领域偏移。`Setophaga_ruticilla` 只有两个多标签正例，其结果没有统计代表性。

结果与数据来源：

- 实验报告：`src/model_train&test/TinyML_model_8class/BirdSet_SSW_evaluation.json`
- 逐片段来源和许可：`dataset_processing/output/MFCC_dataset_BirdSet_SSW_8class/manifest.json`
- 元数据覆盖审计：`row_dataset/BirdSet_SSW/metadata_audit.json`

## 指标与粒度解释

- **Accuracy**：整体预测正确比例，受类别不平衡影响。
- **Balanced Accuracy**：各类别召回率的算术平均，更适合类别不均衡数据。
- **Macro-F1**：先分别计算每类 F1 再等权平均，是当前单标签八类任务的主要综合指标。
- **Top-3**：真实类别是否位于模型概率最高的三个类别中。
- **一秒切片级**：衡量瞬时识别能力，容易受静音、弱鸣声和局部噪声影响。
- **录音/片段级**：平均同一来源连续窗口的 softmax 概率，衡量时间聚合后的识别能力。
- **BirdSet任意目标命中率**：适用于一个片段同时出现多种鸟的情况；其定义不同于单标签 Accuracy。

当前结果表明：模型在内部验证和 DB3V 单鸟/单标签录音上具备中等泛化能力，且时间聚合有效；BirdSet 则揭示了真实多鸟声景、远距离声音和背景噪声下的显著泛化缺口。
