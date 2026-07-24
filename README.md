# 鸟类鸣声分类项目

本项目仅使用 Xeno-canto 鸟鸣录音训练基准分类器。BirdSet SSW 与 DB3V
分别作为跨数据集声景泛化和特定地区泛化的独立测试集，并各自提供与测试录音
隔离的小样本 support 集。当前推荐模型是八类 Xeno-canto Log-Mel DS-CNN；
两个外部 held-out test 均不参与训练、早停或模型选择。

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
│   ├── data_sugment_MFCC.py                # 八类共享预处理核心：MFCC / Log-Mel / PCEN
│   ├── data_sugment_MFCC_DB3V.py           # DB3V 区域数据适配器，支持上述三种特征
│   ├── data_prepare_birdset_ssw.py          # BirdSet SSW 流式筛选及多标签特征预处理
│   ├── data_prepare_birdset_training.py     # BirdSet候选基准诊断切分（多标签）
│   ├── data_prepare_db3v_training.py        # DB3V候选基准诊断切分（同地区留出）
│   ├── birdset_baseline_validation_recordings.json # BirdSet固定验证录音清单
│   ├── prepare_external_fewshot.py          # BirdSet/DB3V support与held-out隔离划分
│   ├── data_sugment_logMel.py              # Log-Mel 预处理入口，共用同一切分逻辑
│   ├── data_sugment_PCEN.py                # PCEN 预处理入口，共用同一切分逻辑
│   ├── label_map_8class.json               # 当前八类连续标签映射
│   └── output/
│       ├── MFCC_dataset_A_8class/          # 当前 Xeno-canto 训练/验证特征
│       ├── MFCC_dataset_DB3V_8class/       # 当前 DB3V 独立测试特征
│       └── MFCC_dataset_BirdSet_SSW_8class/ # 当前 BirdSet 多标签外部特征
└── src/model_train&test/
    ├── Tranin.py                           # 训练入口；按录音级验证 Macro-F1 选择最佳权重
    ├── Train_birdset.py                    # BirdSet多标签候选基准诊断训练
    ├── evaluate_db3v.py                    # DB3V 独立区域评估，输出切片级和录音级指标
    ├── evaluate_birdset_ssw.py             # BirdSet 多标签切片级和片段级评估
    ├── convert.py                           # 可选模型转换入口；不由训练脚本自动调用
    ├── construct_model/
    │   ├── DS_CNN_Model.py
    │   ├── CNN_Model.py
    │   ├── BC_ResNet.py
    │   └── MobileNetV2.py
    ├── TinyML_model_8class/                # MFCC兼容基准模型与报告
    ├── Feature_comparison_8class/          # Xeno三种特征模型及外部测试结果
    ├── BirdSet_feature_comparison_8class/  # BirdSet候选基准诊断结果
    └── DB3V_baseline_diagnostic_8class/    # DB3V候选基准诊断结果
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
- Log-Mel 和 PCEN 的形状均为 `(32, 40)`，录音划分、切片索引及样本数与 MFCC 完全一致，因此特征对比不受数据划分差异影响。
- DB3V 保持为独立地区测试集。评估时会同时报告 1 秒切片指标，以及将一条 DB3V 录音的 8 个 softmax 向量平均后的录音级指标。
- 模型、`*.labels.json` 标签侧车、训练历史、验证摘要与 DB3V JSON/CSV 报告会保存在同一个模型目录中。转换为 TFLite 或 C 头文件时，单独运行 `convert.py`。
- Xeno-canto API 密钥不应写入 README；如需覆盖本地配置，可使用 `XENO_CANTO_API_KEY` 环境变量。音频标准化依赖 FFmpeg。

## 数据来源、许可与实验角色

| 数据 | 上游来源 | 本实验中的角色 | 许可/引用说明 |
|---|---|---|---|
| Xeno-canto | [Xeno-canto](https://xeno-canto.org/) 及其 [API](https://xeno-canto.org/explore/api) | 2,400 条焦点录音用于训练和内部验证 | 每条录音的 Creative Commons 许可保存在对应 `metadata.jsonl`；再分发或商用时必须逐条遵守 |
| DB3V | [Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734) | held-out测试三个地区泛化；隔离support用于后续地区小样本增强 | Zenodo记录标注为 CC BY 4.0；使用时应引用数据集记录 |
| BirdSet | [官方数据集](https://huggingface.co/datasets/DBD-research-group/BirdSet)、[官方代码](https://github.com/DBD-research-group/BirdSet)、[ICLR 2025 论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/484d254ff80e99d543159440a06db0de-Abstract-Conference.html) | SSW held-out测试跨数据集多标签声景泛化；隔离support用于后续小样本增强 | BirdSet代码为 BSD-3-Clause；底层音频仍遵循各来源许可 |
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
4. 完成统一的 16 kHz、单声道、一秒窗口处理，并在相同录音划分和切片上生成 MFCC、Log-Mel、PCEN。
5. 训练八分类 DS-CNN，以录音级 Macro-F1 监控 EarlyStopping 并保存最佳权重。
6. 完成三种特征在 DB3V 三个地区的 1 秒切片级及 8 秒录音级独立评估。
7. 流式读取 BirdSet SSW 全部四个 `test_5s` 分片（约 7.85 GB），未在本地保存完整原始分片。
8. 从 205,200 个 SSW 片段中筛出 19,678 个命中目标物种的五秒多标签片段，分别生成 98,390 个 MFCC、Log-Mel、PCEN 一秒窗口。
9. 将 BirdSet SSW 和 DB3V 分别划分为来源隔离的 few-shot support 与 held-out test，完成最佳 Xeno-canto Log-Mel 模型的两类零样本外部评估。
10. 补做 BirdSet SSW 与 DB3V 分别作为训练来源的反事实候选基准实验；结果仅用于检验其基准适用性，不改变正式数据角色。

尚未完成的实验包括：使用 support 集微调后的增益/遗忘对比、背景/未知类别、多鸟混合增强、量化后精度评估以及其他模型在当前八类规范数据上的公平复测。

## 当前实验角色与完整链路

Xeno-canto 是唯一基准模型训练集。BirdSet SSW 和 DB3V 不再用于从零训练基准
模型，而是并行承担“独立测试”和“小样本增强”两种外部角色：

- **Xeno-canto**：唯一训练/内部验证来源，用于选择特征、模型和早停轮次。
- **BirdSet SSW held-out test**：测试跨数据集、真实多鸟声景和背景噪声泛化。
- **BirdSet SSW support**：仅用于后续少量多标签声景适配，不参与零样本测试。
- **DB3V held-out test**：测试三个特定地区的单标签泛化。
- **DB3V support**：按“地区×类别”抽取，用于后续地区小样本适配。

support 与 held-out test 在原始录音级完全不重叠。任何小样本微调都必须先保存
零样本报告，并且只能使用 support；模型选择仍以 Xeno-canto 内部验证为准，不能
查看 held-out test 调参。

### 为什么BirdSet和DB3V不作为基准训练集

这里的“不能”是实验方法上的角色约束，并不是说两个数据集在技术上无法训练。
如果用它们从头训练模型，它们就会失去当前实验中最重要的独立外部参照作用，
并且无法回答预先设定的跨数据集与跨地区泛化问题。

#### BirdSet SSW

BirdSet SSW 不适合作为本项目八类单标签基准训练集，原因如下：

1. **数据角色不同。** 当前使用的是官方 `test_5s` 多标签声景划分，其设计目标
   是评估真实声景分类，而不是作为本项目的焦点录音训练集。用它训练后就不能再
   将同一批声景作为独立跨数据集测试。
2. **任务定义不同。** 一个五秒片段可以同时包含多个鸟种，需要multi-hot标签和
   sigmoid输出；本项目基准任务是一条焦点录音对应一个主类别的八类softmax。
   从BirdSet训练会同时改变数据集和任务，无法与现有基准作单变量比较。
3. **八类覆盖严重失衡。** 19,678个目标片段来自211条长录音，各类支持差异很大；
   `Setophaga_ruticilla`只有两个片段且来自同一条长录音，无法同时构造无泄漏的
   八类训练集和八类验证集。
4. **一秒窗口存在弱标签噪声。** SSW标签属于完整五秒片段。将标签复制到五个
   一秒窗口时，部分窗口可能没有对应鸟鸣，不适合作为当前一秒单标签基准的直接
   监督信号。
5. **独立来源数量不足。** 98,390个一秒窗口并不等于98,390个独立样本；它们来自
   19,678个相邻片段和211条长录音，空间、设备和背景相关性较强。

因此BirdSet保留为跨数据集、多鸟声景和背景噪声泛化测试；隔离出的support只用于
从Xeno-canto权重开始的小样本适配。下文从BirdSet零初始化的模型是一次性反事实
诊断模型，不纳入正式基准或后续模型选择。

#### DB3V

DB3V不作为基准训练集主要是实验目标和泄漏控制问题：

1. **它是地区泛化的目标域。** 项目需要用DB3V三个地区回答模型在未参与训练的
   地区能否泛化。如果从DB3V训练，测试就退化为同域识别，不能再称为地区泛化。
2. **没有可同时保留的官方基准训练/验证/地区外测协议。** 当前数据按三个地区
   组织；任意从中划分训练集都会让模型接触目标地区的设备、背景和类别先验。
3. **目标域泄漏会高估结果。** 即使训练和测试文件不重复，来自同一地区的录音仍
   共享声学环境。使用DB3V选择特征、阈值或早停轮次，会使区域测试结果带有调参
   偏差。
4. **地区样本量不均衡。** 三个地区分别有5,251、752和4,655条录音，直接混合作为
   基准训练集容易被地区1和地区3主导，并掩盖地区2的不确定性。

因此DB3V held-out只用于特定地区泛化；按“地区×类别”隔离出的120条support只用于
few-shot地区适配。适配后仍必须在剩余10,538条held-out录音上测试，并与零样本
结果比较。

#### Xeno-canto为何满足基准条件

相比之下，当前Xeno-canto子集提供每类300条、总计2,400条焦点录音，八类均衡；
可按完整录音会话稳定划分为每类240条训练和60条验证；标签任务与八类softmax
一致；并且来源ID、音频哈希和DB3V来源已经隔离。它因此能够同时满足类别覆盖、
任务一致、无泄漏模型选择和保留两个外部测试域四项基准要求。

### 三条基准模型链路

| 排名 | 唯一训练集 | 特征 | 输入 | Xeno验证 Macro-F1 | DB3V完整集 Accuracy | DB3V完整集 Macro-F1 | DB3V Top-3 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Xeno-canto | Log-Mel | 32×40 | **63.56%** | **70.71%** | **67.62%** | **91.13%** |
| 2 | Xeno-canto | PCEN | 32×40 | 52.69% | 63.57% | 59.93% | 85.97% |
| 3 | Xeno-canto | MFCC | 32×13 | 49.62% | 56.97% | 50.47% | 84.79% |

Log-Mel 是当前正式推荐链路；PCEN 为噪声归一化研究链路；MFCC 输入最小，保留
为资源受限 TinyML 兼容链路。三者都只能由 Xeno-canto 训练。

### 基准数据集结果与模型选择依据

Xeno-canto 既是唯一训练数据来源，也是唯一允许用于模型选择的数据域。2,400条
录音按完整录音会话固定划分为1,920条训练录音和480条内部验证录音，每类分别为
240/60条；三种特征使用相同录音、相同切片位置、相同DS-CNN、随机种子42、
30轮上限和8轮早停。

| Xeno-canto基准链路 | 验证录音 | Accuracy | Balanced Accuracy | Macro-F1 | 选择结论 |
|---|---:|---:|---:|---:|---|
| Log-Mel + DS-CNN | 480 | **62.29%** | **62.29%** | **63.56%** | 正式推荐模型 |
| PCEN + DS-CNN | 480 | 52.92% | 52.92% | 52.69% | 次选研究模型 |
| MFCC + DS-CNN | 480 | 50.00% | 50.00% | 49.62% | TinyML兼容模型 |

模型选择的唯一主要指标是 **Xeno-canto内部验证录音级 Macro-F1**。每轮先平均
同一验证录音所有一秒窗口的概率，再计算Macro-F1；EarlyStopping和最佳权重恢复
也只监控该指标。因此Log-Mel在查看外部held-out结果之前就已经被选为正式推荐
模型，选择差距如下：

- Log-Mel相对PCEN提高10.87个Macro-F1百分点；
- Log-Mel相对MFCC提高13.94个Macro-F1百分点；
- Accuracy和Balanced Accuracy也由Log-Mel同时取得最高值，没有指标冲突。

BirdSet和DB3V结果不得用于选择特征、早停轮次、阈值或超参数。它们只回答模型
选定后的两个外部问题：是否能跨到真实多标签声景，以及是否能跨到特定地区。
后续few-shot微调也必须保留本表作为原始基准，并报告Xeno验证变化以衡量灾难性
遗忘。

基准结果来源：

- `src/model_train&test/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json`
- `src/model_train&test/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.history.json`
- `src/model_train&test/Feature_comparison_8class/comparison_summary.csv`

### 外部零样本测试与few-shot划分

当前以最佳 `Xeno-canto + Log-Mel + DS-CNN` 模型作为外部测试基准：

| 外部数据 | 测试目的 | support | held-out test | held-out主要结果 |
|---|---|---:|---:|---|
| BirdSet SSW | 跨数据集、多标签真实声景泛化 | 10条长录音，684个片段 | 201条长录音，18,994个片段 | 片段Top-1任意目标17.87%，Top-3 41.29%；纯单物种Macro-F1 9.95% |
| DB3V | 三个特定地区的单标签泛化 | 120条录音 | 10,538条录音 | Accuracy 70.71%，Balanced Accuracy 70.45%，Macro-F1 67.53%，Top-3 91.09% |

BirdSet support 以“每类至少5个正片段”为目标，但必须保持长录音完整，所以实际
片段数会超过40；稀有的 `Setophaga_ruticilla` 只有同一长录音中的两个片段，
无法达到5-shot。DB3V support 则在每个地区、每个类别各抽取5条录音，共120条。

这两个零样本结果是后续小样本增强的固定前测。微调后必须同时复测：

1. 对应 held-out test，衡量目标数据域提升；
2. Xeno-canto 内部验证，检查是否发生灾难性遗忘；
3. 另一个外部 held-out test，检查增强是否损害跨域泛化。

结果与划分来源：

- `dataset_processing/output/LogMel_BirdSet_external_split_8class/split_manifest.json`
- `dataset_processing/output/LogMel_DB3V_external_split_8class/split_manifest.json`
- `src/model_train&test/Feature_comparison_8class/LogMel/BirdSet_SSW_heldout_evaluation.json`
- `src/model_train&test/Feature_comparison_8class/LogMel/DB3V_heldout_evaluation.json`
- `src/model_train&test/Feature_comparison_8class/LogMel/DB3V_heldout_evaluation_summary.csv`

### 候选基准数据集反事实诊断（不参与正式选型）

为验证前述数据角色判断，项目另外执行了两组从零训练实验：分别把BirdSet SSW和
DB3V当作候选基准训练来源，再到另一个完全独立的数据集测试。该实验只回答“如果
强行将其作为基准会发生什么”，其内部验证结果不参与正式模型选择，也不会改变
Xeno-canto的唯一正式基准地位。

#### BirdSet作为候选基准

BirdSet按原始长录音固定拆分为169条训练录音（16,102个五秒片段）和42条验证录音
（3,576个片段），使用multi-hot、sigmoid和二元交叉熵。由于
`Setophaga_ruticilla`只有同一长录音中的两个片段，该录音只能保留在训练集，
所以内部Macro-F1只覆盖验证集中有正样本的7类。外部测试使用未参与训练和选型的
完整DB3V 10,658条录音。

| 特征 | BirdSet验证Top-1任意目标 | BirdSet验证7类Macro-F1 | DB3V录音Accuracy | DB3V录音Balanced Accuracy | DB3V录音Macro-F1 | 同特征Xeno→DB3V Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| MFCC | 31.46% | **14.86%** | **17.72%** | 13.97% | 9.57% | 50.47% |
| Log-Mel | **46.59%** | 14.77% | 15.73% | **17.44%** | **9.97%** | **67.62%** |
| PCEN | 17.81% | 0.00% | 11.04% | 9.64% | 4.52% | 59.93% |

PCEN的0.00%表示在固定0.5 sigmoid阈值下没有形成有效的验证集正类预测，不表示
argmax完全随机；其Top-1任意目标仍为17.81%。更关键的是，三种BirdSet训练模型
跨到DB3V后的Macro-F1仅4.52%–9.97%，比相同特征的Xeno-canto模型低
40.90–57.65个百分点。结果同时暴露了七类验证、任务定义不一致、长录音来源少和
跨域退化，因此BirdSet不能承担当前八类单标签正式基准。

#### DB3V作为候选基准

DB3V按“地区×类别”在完整八秒录音级拆分：8,526条训练、2,132条验证；三个地区
都同时出现在两部分中。内部指标因此只是**同地区留出识别**，不是未见地区泛化。
跨数据集测试使用完整BirdSet SSW目标子集，共211条长录音、19,678个五秒片段；
“纯单物种Macro-F1”在10,230个全局单物种片段上计算。

| 特征 | DB3V验证Accuracy | DB3V验证Balanced Accuracy | DB3V验证Macro-F1 | BirdSet Top-1任意目标 | BirdSet Top-3任意目标 | BirdSet纯单物种Macro-F1 | 同特征Xeno→BirdSet纯单物种Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCC | 88.65% | 90.16% | 88.44% | 15.19% | 51.98% | 11.32% | 9.77% |
| Log-Mel | **91.56%** | **91.69%** | **90.33%** | **19.79%** | **53.45%** | **15.81%** | **10.12%** |
| PCEN | 84.85% | 82.80% | 82.17% | 16.43% | 36.96% | 14.00% | 8.08% |

DB3V Log-Mel在同地区验证上达到90.33% Macro-F1，并在同口径BirdSet全集上略高于
同特征Xeno模型。这说明DB3V可提供有价值的目标域监督，适合后续地区support微调；
但高内部得分主要回答“见过三个目标地区后能否识别同地区新文件”。一旦用DB3V
训练，就失去DB3V作为独立地区泛化测试的能力，也无法再回答模型是否泛化到未参与
训练的地区。因此该结果支持“DB3V适合目标域适配、不适合作为正式基准”的定位。

综合两组诊断：BirdSet训练后跨DB3V严重退化；DB3V训练虽能改善部分BirdSet指标，
却消耗了关键地区测试域。只有Xeno-canto同时保留八类单标签一致性、无泄漏内部
选型和两个独立外部数据域，所以正式基准及推荐链路保持不变。

诊断实验来源：

- BirdSet固定切分与协议：`dataset_processing/birdset_baseline_validation_recordings.json`、`src/model_train&test/BirdSet_feature_comparison_8class/experiment_protocol.json`
- BirdSet训练结果：`src/model_train&test/BirdSet_feature_comparison_8class/comparison_summary.csv`、`src/model_train&test/BirdSet_baseline_8class/`
- DB3V固定协议与结果：`src/model_train&test/DB3V_baseline_diagnostic_8class/experiment_protocol.json`、`src/model_train&test/DB3V_baseline_diagnostic_8class/comparison_summary.csv`
- DB3V各特征原始报告：`src/model_train&test/DB3V_baseline_diagnostic_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json` 与 `BirdSet_SSW_evaluation.json`
- 同口径Xeno参考：`src/model_train&test/Feature_comparison_8class/birdset_full_summary.csv` 与 `{MFCC,LogMel,PCEN}/BirdSet_SSW_full_evaluation.json`
- 数据集原始出处见上文“数据来源、引用与许可”；诊断协议文件也保存了BirdSet、SSW和DB3V的官方链接。

## 当前 DS-CNN 基准模型

当前兼容基准模型为 MFCC `DS_CNN_Model`；最高精度推荐模型是
`Feature_comparison_8class/LogMel/DS_CNN_Model.h5`：

- 推荐模型输入：一秒音频对应的 `(32, 40, 1)` Log-Mel；
- 兼容模型输入：一秒音频对应的 `(32, 13, 1)` MFCC；
- 输出：八类 softmax；
- 训练数据：Xeno-canto；
- 独立测试：BirdSet SSW 与 DB3V；两者另有完全隔离的小样本 support；
- 推荐模型文件：`src/model_train&test/Feature_comparison_8class/LogMel/DS_CNN_Model.h5`；
- 兼容模型文件：`src/model_train&test/TinyML_model_8class/DS_CNN_Model.h5`。

内部留出验证以完整录音为评估单位：

| 数据与粒度 | 数量 | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Xeno-canto内部验证，录音级 | 480条录音 | 50.00% | 50.00% | **49.62%** |

结果来源：`src/model_train&test/TinyML_model_8class/DS_CNN_Model.validation.json`。

## 八分类特征处理方法对比

`dataset_processing` 中与本实验有关的四个入口已整理为两类职责：

1. `data_sugment_MFCC.py` 是共享实现，可生成 MFCC、Log-Mel 或 PCEN；
2. `data_sugment_logMel.py` 和 `data_sugment_PCEN.py` 是调用共享实现的便捷入口；
3. `data_sugment_MFCC_DB3V.py` 是 DB3V 区域数据适配器，不是第四种声学特征，可通过 `--feature` 对三种特征执行一致的外部测试预处理。

公平复测固定使用相同的八类标签顺序、2,400 条 Xeno-canto 原始录音、1,920/480 条录音级训练/验证划分、每条录音最多 8 个相同位置的一秒切片、DS-CNN、随机种子 42、最多 30 轮和 8 轮早停。DB3V 仅用于最终测试，不参与模型选择。

| 特征 | 输入形状 | Xeno-canto 验证 Accuracy | 验证 Macro-F1 | DB3V 切片 Accuracy | DB3V 切片 Macro-F1 | DB3V 录音 Accuracy | DB3V 录音 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCC | 32×13 | 50.00% | 49.62% | 39.80% | 35.48% | 56.97% | 50.47% |
| Log-Mel | 32×40 | **62.29%** | **63.56%** | **44.67%** | **42.84%** | **70.71%** | **67.62%** |
| PCEN | 32×40 | 52.92% | 52.69% | 40.03% | 37.71% | 63.57% | 59.93% |

本轮中 Log-Mel 最优：相对 MFCC，内部验证 Macro-F1 提高 13.94 个百分点，DB3V 录音级 Macro-F1 提高 17.15 个百分点；PCEN 也分别提高 3.07 和 9.46 个百分点。该结论只适用于本次固定 DS-CNN 和当前数据协议，还不能直接外推到其他模型或噪声条件。

复现实验时分别运行三个 Xeno-canto 入口，并为 DB3V 入口传入匹配的 `--feature`。训练和测试命令与“推荐工作流”相同，只需对应修改数据集和模型目录。

结果来源：

- `src/model_train&test/Feature_comparison_8class/comparison_summary.csv`
- `src/model_train&test/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json`
- `src/model_train&test/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DB3V_evaluation.json`
- `src/model_train&test/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DB3V_evaluation_summary.csv`

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

## BirdSet与DB3V外部测试及小样本协议

BirdSet 和 DB3V 均不用于训练基准模型。推荐使用与最佳 Xeno-canto 模型匹配的
Log-Mel 特征，先建立 support/test 隔离划分，再保存零样本报告。

```powershell
# BirdSet：按原始长录音拆分 support 与 held-out test
& .\.venv\Scripts\python.exe dataset_processing\prepare_external_fewshot.py `
  birdset `
  --source-dir dataset_processing\output\LogMel_dataset_BirdSet_SSW_8class `
  --output-dir dataset_processing\output\LogMel_BirdSet_external_split_8class `
  --shots 5

# DB3V：按地区和类别拆分 support 与 held-out test
& .\.venv\Scripts\python.exe dataset_processing\prepare_external_fewshot.py `
  db3v `
  --source-dir dataset_processing\output\LogMel_dataset_DB3V_8class `
  --output-dir dataset_processing\output\LogMel_DB3V_external_split_8class `
  --shots 5

# BirdSet held-out：跨数据集、多标签声景泛化
& .\.venv\Scripts\python.exe "src\model_train&test\evaluate_birdset_ssw.py" `
  --models DS_CNN_Model `
  --dataset-dir dataset_processing\output\LogMel_BirdSet_external_split_8class\test `
  --model-dir "src\model_train&test\Feature_comparison_8class\LogMel" `
  --output "src\model_train&test\Feature_comparison_8class\LogMel\BirdSet_SSW_heldout_evaluation.json"

# DB3V held-out：三个特定地区泛化
& .\.venv\Scripts\python.exe "src\model_train&test\evaluate_db3v.py" `
  --models DS_CNN_Model `
  --dataset-dir dataset_processing\output\LogMel_DB3V_external_split_8class `
  --model-dir "src\model_train&test\Feature_comparison_8class\LogMel" `
  --output "src\model_train&test\Feature_comparison_8class\LogMel\DB3V_heldout_evaluation.json"
```

### 小样本增强实验约束

- support 只能用于从 Xeno-canto 基准权重开始的微调，不能从头训练新基准。
- BirdSet support 保留 multi-hot 标签；DB3V support 使用单标签。
- held-out test 永远不参与梯度更新、阈值选择、早停或超参数选择。
- 必须报告 zero-shot 与 few-shot 的差值，而不是只报告微调后最高值。
- BirdSet 和 DB3V 应分别微调，才能区分“跨数据集适配”与“特定地区适配”。
- 每次微调后同时复测 Xeno-canto、BirdSet held-out 和 DB3V held-out。

## 指标与粒度解释

- **Accuracy**：整体预测正确比例，受类别不平衡影响。
- **Balanced Accuracy**：各类别召回率的算术平均，更适合类别不均衡数据。
- **Macro-F1**：先分别计算每类 F1 再等权平均，是当前单标签八类任务的主要综合指标。
- **Top-3**：真实类别是否位于模型概率最高的三个类别中。
- **一秒切片级**：衡量瞬时识别能力，容易受静音、弱鸣声和局部噪声影响。
- **录音/片段级**：平均同一来源连续窗口的 softmax 概率，衡量时间聚合后的识别能力。
- **BirdSet任意目标命中率**：用于 BirdSet 内部多标签验证，表示概率最高类别是否命中任一真实标签；其定义不同于单标签 Accuracy。

当前结果表明：Xeno-canto Log-Mel 主模型在 DB3V 三地区保持较强泛化，但在
BirdSet 多鸟声景上的 Top-1 任意目标命中率仅为 17.87%。因此后续增强实验应
分别回答两个问题：BirdSet support 能否改善跨数据集声景泛化；DB3V support
能否进一步改善特定地区泛化，同时不损害另一个外部数据域。
