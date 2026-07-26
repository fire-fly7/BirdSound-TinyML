# 鸟类鸣声分类项目

本项目仅使用 Xeno-canto 鸟鸣录音训练基准分类器。BirdSet SSW 与 DB3V
分别作为跨数据集声景泛化和特定地区泛化的独立测试集，并各自提供与测试录音
隔离的小样本 support 集。当前推荐模型是八类 Xeno-canto Log-Mel DS-CNN；
两个外部 held-out test 均不参与训练、早停或模型选择。BirdSet grouped 5-shot
多标签适配和 DB3V 5/10/20-shot 单标签适配均已完成。

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

当前原始集位于 `src/row_dataset/row_bird_dataset_A_next/`，共 2,400 条录音。每个文件均为 16 kHz、单声道、16-bit PCM WAV；全局 ID 与音频哈希已去重，并与 DB3V 来源 ID 隔离。

## 项目结构

```text
.
├── README.md
└── src/
    ├── README.md                           # src目录职责与Git边界
    ├── row_dataset/
    │   ├── data_crawler_xeno_canto.py       # Xeno-canto 爬虫；支持八类、A–E、DB3V 来源排除
    │   ├── data_crawler_DB3V.py             # DB3V 原始数据下载/整理
    │   ├── DB3V/
    │   │   └── extracted/data_wav_8s_2/     # 原始 DB3V WAV；仅用于独立评估/后续小样本实验
    │   ├── BirdSet_SSW/                     # BirdSet SSW 元数据缓存；原始分片不落盘
    │   ├── row_bird_dataset_A_next/         # 当前八类 Xeno-canto 原始训练数据
    │   └── xeno_removed_db3v_overlap.json   # 历史 DB3V/Xeno-canto 来源重叠记录
    ├── dataset_processing/
    │   ├── data_sugment_MFCC.py             # 八类共享预处理核心：MFCC / Log-Mel / PCEN
    │   ├── data_sugment_MFCC_DB3V.py        # DB3V 区域数据适配器，支持上述三种特征
    │   ├── data_prepare_birdset_ssw.py       # BirdSet SSW 流式筛选及多标签特征预处理
    │   ├── data_prepare_birdset_training.py  # BirdSet候选基准诊断切分（多标签）
    │   ├── data_prepare_db3v_training.py     # DB3V候选基准诊断切分（同地区留出）
    │   ├── prepare_external_fewshot.py       # BirdSet/DB3V support与held-out隔离划分
    │   ├── label_map_8class.json            # 当前八类连续标签映射
    │   └── output/                          # 生成特征；不进入Git
    └── experiments/
        ├── README.md                        # 实验目录与数据角色说明
        ├── Tranin.py                        # 单标签训练入口
        ├── Train_birdset.py                 # BirdSet多标签诊断训练
        ├── fine_tune_birdset.py             # BirdSet grouped 5-shot多标签适配
        ├── fine_tune_db3v.py                # DB3V 5/10/20-shot地区适配
        ├── run_db3v_ablation_multiseed.py   # DB3V严格策略多随机种子实验与汇总
        ├── int8_inference.py                # 特征感知严格INT8接口与批量推理
        ├── evaluate_int8_experiments.py     # 零样本/小样本INT8统一复测
        ├── evaluate_db3v.py                 # DB3V区域评估
        ├── evaluate_birdset_ssw.py          # BirdSet多标签评估
        ├── summarize_birdset_fewshot.py     # BirdSet结果汇总重建
        ├── convert.py                       # 模型转换入口
        ├── construct_model/                 # 模型结构
        ├── TinyML_model_8class/             # MFCC兼容基准模型与报告
        ├── Feature_comparison_8class/       # Xeno三特征正式实验
        ├── BirdSet_fewshot_8class/          # BirdSet小样本模型与跨域复测
        ├── DB3V_fewshot_ablation_multiseed_8class/ # 108组严格策略实验与统计
        ├── INT8_quantization_8class/         # 15条链路的INT8模型与精度报告
        ├── BirdSet_baseline_8class/         # BirdSet MFCC诊断结果
        ├── BirdSet_feature_comparison_8class/
        └── DB3V_baseline_diagnostic_8class/
```

目录树只列出当前八类流程的规范路径。并行抓取暂存目录、旧实验工件、`__pycache__/`、`.venv/` 和日志文件都不是当前训练输入或源码结构。

## 推荐工作流

使用项目虚拟环境执行以下命令。重新抓取时不要使用 `--allow-partial`，以确保每个类别满足 300 条有效录音的下限。

```powershell
# 1. 获取八类 Xeno-canto 原始数据（默认接受 A,B,C,D,E 评分）
& .\.venv\Scripts\python.exe src\row_dataset\data_crawler_xeno_canto.py `
  --eligible-eight `
  --target-per-species 300 `
  --min-per-species 300 `
  --output-dir src\row_dataset\row_bird_dataset_A_next

# 2. 生成 Xeno-canto 训练/验证 MFCC 特征
& .\.venv\Scripts\python.exe src\dataset_processing\data_sugment_MFCC.py `
  --input-dir src\row_dataset\row_bird_dataset_A_next `
  --output-dir src\dataset_processing\output\MFCC_dataset_A_8class

# 3. 生成同标签顺序的 DB3V 独立测试特征
& .\.venv\Scripts\python.exe src\dataset_processing\data_sugment_MFCC_DB3V.py `
  --output-dir src\dataset_processing\output\MFCC_dataset_DB3V_8class `
  --xeno-label-map src\dataset_processing\output\MFCC_dataset_A_8class\label_map.json

# 4. 训练 DS-CNN；EarlyStopping 监控录音级 Macro-F1
$env:TF_CPP_MIN_LOG_LEVEL = '2'
& .\.venv\Scripts\python.exe "src\experiments\Tranin.py" `
  --model DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\MFCC_dataset_A_8class `
  --model-dir "src\experiments\TinyML_model_8class" `
  --epochs 30 --patience 8

# 5. 在完全独立的 DB3V 上测试
& .\.venv\Scripts\python.exe "src\experiments\evaluate_db3v.py" `
  --models DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\MFCC_dataset_DB3V_8class `
  --model-dir "src\experiments\TinyML_model_8class"
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
| DB3V | [Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734) | held-out测试三个地区泛化；隔离support已用于5/10/20-shot地区适配 | Zenodo记录标注为 CC BY 4.0；使用时应引用数据集记录 |
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

截至 **2026-07-27**，已经完成：

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
11. 对 MFCC、LogMel、PCEN 三条 Xeno-canto DS-CNN 链路完成 DB3V
    5/10/20-shot 地区适配；每个shot比较 `head`、`last_block`、`all` 三种策略，
    共训练和记录27组候选。
12. 建立固定随机种子42的嵌套support：120条 ⊂ 239条 ⊂ 461条；针对DB3V
    稀疏分层明确记录10-shot缺1条、20-shot缺19条，不使用重复录音补齐。
13. 在共同的10,197条DB3V held-out录音上统一复测零样本与5/10/20-shot模型，
    并同步复测Xeno-canto内部验证和完整BirdSet SSW，完成目标域增益、基准保留率
    与跨数据集代价对比。
14. 原27组单种子历史实验的综合链路为
    `Xeno-canto → LogMel → DS-CNN → DB3V 20-shot all微调（BatchNorm冻结）`；
    若只优化DB3V地区性能，则PCEN 20-shot `last_block`取得最高Macro-F1 70.60%。
15. 对 MFCC、LogMel、PCEN 三条 Xeno-canto DS-CNN 链路完成 BirdSet grouped
    5-shot 多标签适配，共训练9组策略；在隔离的201条长录音上复测，并同步评估
    Xeno-canto和完整DB3V。PCEN分类头微调在三个域上均取得正增益。
16. 完成3条零样本、9条DB3V小样本和3条BirdSet小样本选中链路的严格INT8
    后训练量化与全量精度复测；15个模型均为int8输入/输出、0个浮点张量。
    MFCC最稳定，LogMel是绝对精度与稳定性的折中，PCEN出现严重量化退化。
17. 补齐严格 `Head-Only`、`BN+Head`、`BN+Head+Replay` 和包含BatchNorm的
    `Full Fine-Tuning`；在三种特征、5/10/20-shot和随机种子42/123/2026上完成
    108组训练、Xeno-canto遗忘测试和共同DB3V held-out评估，并报告均值与样本标准差。

尚未完成的实验包括：背景/未知类别、多鸟混合增强、PCEN量化感知训练、其他模型在
当前八类规范数据上的公平复测，以及改变外部support抽样的重复实验和置信区间估计。

## 当前实验角色与完整链路

Xeno-canto 是唯一基准模型训练集。BirdSet SSW 和 DB3V 不再用于从零训练基准
模型，而是并行承担“独立测试”和“小样本增强”两种外部角色：

- **Xeno-canto**：唯一训练/内部验证来源，用于选择特征、模型和早停轮次。
- **BirdSet SSW held-out test**：测试跨数据集、真实多鸟声景和背景噪声泛化。
- **BirdSet SSW support**：已完成 grouped 5-shot 多标签声景适配，不参与零样本测试。
- **DB3V held-out test**：测试三个特定地区的单标签泛化。
- **DB3V support**：按“地区×类别”抽取；已完成5/10/20-shot地区小样本适配。

support 与 held-out test 在原始录音级完全不重叠。任何小样本微调都必须先保存
零样本报告，并且只能使用support。DB3V使用support内部录音级Macro-F1和
Xeno-canto保留率联合选型；BirdSet使用support内部片段Top-1任意目标命中率和
Xeno-canto保留率联合选型。两个held-out均不用于选择策略、轮次或学习率。

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

因此DB3V held-out只用于特定地区泛化。项目先保留120条support/10,538条测试录音
的原始5-shot报告，再补充嵌套的120/239/461条support实验；跨shot比较统一使用
20-shot support之外的10,197条共同held-out录音，并与相同测试集上的零样本结果
比较。

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

- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.history.json`
- `src/experiments/Feature_comparison_8class/comparison_summary.csv`

### 外部零样本测试与few-shot划分

当前以最佳 `Xeno-canto + Log-Mel + DS-CNN` 模型作为外部测试基准：

| 外部数据 | 测试目的 | support | held-out test | held-out主要结果 |
|---|---|---:|---:|---|
| BirdSet SSW | 跨数据集、多标签真实声景泛化 | 10条长录音，684个片段 | 201条长录音，18,994个片段 | 片段Top-1任意目标17.87%，Top-3 41.29%；纯单物种Macro-F1 9.95% |
| DB3V | 三个特定地区的单标签泛化 | 120条录音 | 10,538条录音 | Accuracy 70.71%，Balanced Accuracy 70.45%，Macro-F1 67.53%，Top-3 91.09% |

BirdSet support 以“每类至少5个正片段”为目标，但必须保持长录音完整，所以实际
片段数会超过40；稀有的 `Setophaga_ruticilla` 只有同一长录音中的两个片段，
无法达到5-shot。DB3V已进一步构建嵌套的目标5/10/20-shot support，实际分别为
120、239和461条录音；横向比较使用共同的10,197条held-out录音。

这两个零样本结果是小样本增强的固定前测。DB3V与BirdSet support适配均已完成；
每次微调后均按以下三个方向复测：

1. 对应 held-out test，衡量目标数据域提升；
2. Xeno-canto 内部验证，检查是否发生灾难性遗忘；
3. 另一个外部 held-out test，检查增强是否损害跨域泛化。

结果与划分来源：

- `src/dataset_processing/output/LogMel_BirdSet_external_split_8class/split_manifest.json`
- `src/dataset_processing/output/LogMel_DB3V_external_split_8class/split_manifest.json`
- `src/experiments/Feature_comparison_8class/LogMel/BirdSet_SSW_heldout_evaluation.json`
- `src/experiments/Feature_comparison_8class/LogMel/DB3V_heldout_evaluation.json`
- `src/experiments/Feature_comparison_8class/LogMel/DB3V_heldout_evaluation_summary.csv`

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

- BirdSet固定切分与协议：`src/dataset_processing/birdset_baseline_validation_recordings.json`、`src/experiments/BirdSet_feature_comparison_8class/experiment_protocol.json`
- BirdSet训练结果：`src/experiments/BirdSet_feature_comparison_8class/comparison_summary.csv`、`src/experiments/BirdSet_baseline_8class/`
- DB3V固定协议与结果：`src/experiments/DB3V_baseline_diagnostic_8class/experiment_protocol.json`、`src/experiments/DB3V_baseline_diagnostic_8class/comparison_summary.csv`
- DB3V各特征原始报告：`src/experiments/DB3V_baseline_diagnostic_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json` 与 `BirdSet_SSW_evaluation.json`
- 同口径Xeno参考：`src/experiments/Feature_comparison_8class/birdset_full_summary.csv` 与 `{MFCC,LogMel,PCEN}/BirdSet_SSW_full_evaluation.json`
- 数据集原始出处见上文“数据来源、引用与许可”；诊断协议文件也保存了BirdSet、SSW和DB3V的官方链接。

## 当前 DS-CNN 基准模型

当前兼容基准模型为 MFCC `DS_CNN_Model`；最高精度推荐模型是
`Feature_comparison_8class/LogMel/DS_CNN_Model.h5`：

- 推荐模型输入：一秒音频对应的 `(32, 40, 1)` Log-Mel；
- 兼容模型输入：一秒音频对应的 `(32, 13, 1)` MFCC；
- 输出：八类 softmax；
- 训练数据：Xeno-canto；
- 独立测试：BirdSet SSW 与 DB3V；两者另有完全隔离的小样本 support；
- 推荐模型文件：`src/experiments/Feature_comparison_8class/LogMel/DS_CNN_Model.h5`；
- 兼容模型文件：`src/experiments/TinyML_model_8class/DS_CNN_Model.h5`。

内部留出验证以完整录音为评估单位：

| 数据与粒度 | 数量 | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Xeno-canto内部验证，录音级 | 480条录音 | 50.00% | 50.00% | **49.62%** |

结果来源：`src/experiments/TinyML_model_8class/DS_CNN_Model.validation.json`。

## 八分类特征处理方法对比

`src/dataset_processing` 中与本实验有关的四个入口已整理为两类职责：

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

- `src/experiments/Feature_comparison_8class/comparison_summary.csv`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DB3V_evaluation.json`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DB3V_evaluation_summary.csv`

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

- `src/experiments/TinyML_model_8class/DB3V_evaluation.json`
- `src/experiments/TinyML_model_8class/DB3V_evaluation_summary.csv`

## BirdSet与DB3V外部测试及小样本协议

BirdSet 和 DB3V 均不用于训练基准模型。推荐使用与最佳 Xeno-canto 模型匹配的
Log-Mel 特征，先建立 support/test 隔离划分，再保存零样本报告。

```powershell
# BirdSet：按原始长录音拆分 support 与 held-out test
& .\.venv\Scripts\python.exe src\dataset_processing\prepare_external_fewshot.py `
  birdset `
  --source-dir src\dataset_processing\output\LogMel_dataset_BirdSet_SSW_8class `
  --output-dir src\dataset_processing\output\LogMel_BirdSet_external_split_8class `
  --shots 5

# DB3V：按地区和类别拆分 support 与 held-out test
& .\.venv\Scripts\python.exe src\dataset_processing\prepare_external_fewshot.py `
  db3v `
  --source-dir src\dataset_processing\output\LogMel_dataset_DB3V_8class `
  --output-dir src\dataset_processing\output\LogMel_DB3V_external_split_8class `
  --shots 5

# BirdSet held-out：跨数据集、多标签声景泛化
& .\.venv\Scripts\python.exe "src\experiments\evaluate_birdset_ssw.py" `
  --models DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\LogMel_BirdSet_external_split_8class\test `
  --model-dir "src\experiments\Feature_comparison_8class\LogMel" `
  --output "src\experiments\Feature_comparison_8class\LogMel\BirdSet_SSW_heldout_evaluation.json"

# DB3V held-out：三个特定地区泛化
& .\.venv\Scripts\python.exe "src\experiments\evaluate_db3v.py" `
  --models DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\LogMel_DB3V_external_split_8class `
  --model-dir "src\experiments\Feature_comparison_8class\LogMel" `
  --output "src\experiments\Feature_comparison_8class\LogMel\DB3V_heldout_evaluation.json"
```

### 小样本增强实验约束

- support 只能用于从 Xeno-canto 基准权重开始的微调，不能从头训练新基准。
- BirdSet support 保留 multi-hot 标签；DB3V support 使用单标签。
- held-out test 永远不参与梯度更新、阈值选择、早停或超参数选择。
- 必须报告 zero-shot 与 few-shot 的差值，而不是只报告微调后最高值。
- BirdSet 和 DB3V 应分别微调，才能区分“跨数据集适配”与“特定地区适配”。
- 每次微调后同时复测 Xeno-canto、BirdSet held-out 和 DB3V held-out。

### DB3V 5/10/20-shot 地区适配结果

已对 Xeno-canto 训练的 MFCC、LogMel、PCEN 三条 DS-CNN 链路完成 DB3V
5/10/20-shot 小样本优化。三个 support 集固定随机种子42并嵌套为
`5-shot ⊂ 10-shot ⊂ 20-shot`；三种特征使用完全相同的录音身份。由于地区2的
`Setophaga_aestiva` 只有9条录音、`Certhia_americana` 只有12条录音，目标
10/20-shot不能在所有“地区×类别”分层中严格满足。实验不做重复采样，而是按唯一
录音的可用上限截断，实际 support 数为120、239和461条。

#### 实验记录

| 项目 | 固定配置 |
|---|---|
| 执行时间 | 2026-07-26（Asia/Tokyo；原始 JSON 使用 UTC 时间戳） |
| 基准权重 | `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.h5` |
| 类别与任务 | 8 类单标签、softmax、categorical cross-entropy |
| 随机性 | TensorFlow/NumPy 随机种子均为 42 |
| support | 5-shot：120条/960窗口；目标10-shot：239条/1,912窗口；目标20-shot：461条/3,688窗口 |
| 稀疏分层缺口 | 10-shot缺1条；20-shot缺19条；均不使用重复录音补齐 |
| 选型切分 | 每分层按80/20拆分：5-shot为4+1，10-shot为8+2，20-shot通常为16+4；实际训练/验证为96/24、191/48、365/96条 |
| 共同 held-out test | 固定使用20-shot support之外的10,197条录音、81,576个窗口；所有基准与5/10/20-shot模型使用同一测试集 |
| 优化器与批量 | Adam，batch size 32 |
| 最大轮次与早停 | 最多30轮；选型分数连续6轮不改善即停止并恢复最佳权重 |
| 微调策略 | `head`：分类头，LR 3e-4；`last_block`：末端可分离卷积块和分类头，LR 1e-4；`all`：除BatchNorm外全网络，LR 3e-5 |
| 运行环境 | Python 3.11.9、TensorFlow 2.19.0、NumPy 2.1.3 |

模型选择时，各shot规模都按“地区×类别”分层固定拆分约80%训练、20%验证。
分别比较分类头、最后一个深度可分离卷积块和全网络微调；
选择指标为“support 验证录音级 Macro-F1 × 截断到 1 的 Xeno-canto
Macro-F1 保留率”。选择结束后重新加载原始 Xeno-canto 权重，使用对应规模的全部
support 按选定轮次训练。共同DB3V held-out在全部选型结束前未被读取。

5-shot的九组策略选型记录如下。这里的 Xeno 指标是选型阶段最佳权重的结果；最终
使用全部 support 重训后的 Xeno 结果见下一张选中策略表。

| 特征 | 策略 | LR | 最佳轮次 | support验证Macro-F1 | 选型阶段Xeno Macro-F1 | Xeno保留率 | 选型分数 | 采用 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MFCC | head | 3e-4 | 9 | 65.83% | 47.69% | 96.12% | **63.28%** | 是 |
| MFCC | last_block | 1e-4 | 10 | **69.73%** | 44.69% | 90.06% | 62.80% | 否 |
| MFCC | all | 3e-5 | 2 | 45.36% | **50.43%** | 101.64% | 45.36% | 否 |
| LogMel | head | 3e-4 | 11 | **63.93%** | 61.83% | 97.28% | **62.19%** | 是 |
| LogMel | last_block | 1e-4 | 1 | 58.17% | **63.24%** | 99.50% | 57.89% | 否 |
| LogMel | all | 3e-5 | 4 | 57.68% | 62.43% | 98.22% | 56.65% | 否 |
| PCEN | head | 3e-4 | 5 | 79.73% | **60.92%** | **115.62%** | 79.73% | 否 |
| PCEN | last_block | 1e-4 | 5 | 79.73% | 60.67% | 115.15% | 79.73% | 否 |
| PCEN | all | 3e-5 | 4 | **83.42%** | 60.36% | 114.55% | **83.42%** | 是 |

| 特征 | 选中策略 | 轮次 | support验证Macro-F1 | 选型分数 | Xeno Macro-F1（前→后） | 保留率 |
|---|---|---:|---:|---:|---:|---:|
| MFCC | 仅分类头 | 9 | 65.83% | 63.28% | 49.62% → 46.10% | 92.90% |
| LogMel | 仅分类头 | 11 | 63.93% | 62.19% | 63.56% → 60.05% | 94.48% |
| PCEN | 全网络，BatchNorm冻结 | 4 | **83.42%** | **83.42%** | 52.69% → 59.92% | **113.72%** |

DB3V 留出集结果以八秒录音为粒度，先平均每条录音的八个一秒窗口 softmax：

| 特征 | Accuracy（前→后） | Balanced Accuracy（前→后） | Macro-F1（前→后） | Macro-F1变化 | Top-3（前→后） |
|---|---:|---:|---:|---:|---:|
| MFCC | 57.00% → 61.06% | 52.06% → 57.79% | 50.40% → 55.96% | **+5.55 pp** | 84.78% → 86.57% |
| LogMel | 70.71% → **72.25%** | 70.45% → **71.10%** | 67.53% → **67.95%** | +0.42 pp | 91.09% → **92.10%** |
| PCEN | 63.58% → 71.13% | 63.17% → 69.47% | 59.87% → 66.14% | **+6.27 pp** | 85.94% → 90.90% |

三条链路都提高了 DB3V held-out Macro-F1。PCEN 的增幅最大，且 Xeno-canto
验证也同步提高；LogMel 微调后的绝对 Macro-F1 仍最高，但由于其零样本基准已经
很强，5-shot 的边际增益只有 0.42 个百分点。

为检查地区适配是否损害另一个数据域，使用未参与训练和选型的完整 BirdSet SSW
目标子集（211 条长录音、19,678 个片段）复测：

| 特征 | Top-1任意目标（前→后） | Top-3任意目标（前→后） | 纯单物种Macro-F1（前→后） |
|---|---:|---:|---:|
| MFCC | 17.32% → 10.72% | 44.33% → 46.58% | 9.77% → 7.00% |
| LogMel | 18.27% → **18.81%** | 41.56% → **48.66%** | 10.12% → 9.19% |
| PCEN | 19.57% → 9.81% | 26.22% → 29.08% | 8.08% → 9.42% |

以上是最初5-shot实验在其10,538条专用held-out上的结果，保留用于追溯。下面的
5/10/20-shot横向结论全部改用相同的10,197条共同held-out，不能把两种测试规模的
数值直接混合比较。由于20-shot已用完地区2中 `Certhia_americana` 和
`Setophaga_aestiva` 的全部录音，共同测试集在地区2不再包含这两个分层；汇总测试
仍通过地区1和3覆盖全部八类。

#### 10/20-shot候选策略选型记录

| Shot | 特征 | 策略 | 最佳轮次 | support验证Macro-F1 | 选型Xeno Macro-F1 | Xeno保留率 | 选型分数 | 采用 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 10 | MFCC | head | 5 | **72.01%** | 47.93% | 96.60% | **69.56%** | 是 |
| 10 | MFCC | last_block | 4 | 67.93% | 46.89% | 94.50% | 64.19% | 否 |
| 10 | MFCC | all | 4 | 66.20% | **48.37%** | **97.49%** | 64.54% | 否 |
| 10 | LogMel | head | 1 | **83.62%** | 61.96% | 97.48% | 81.51% | 否 |
| 10 | LogMel | last_block | 2 | 83.42% | 62.23% | 97.91% | **81.68%** | 是 |
| 10 | LogMel | all | 2 | 79.11% | **62.93%** | **99.01%** | 78.33% | 否 |
| 10 | PCEN | head | 1 | **74.86%** | 58.87% | 111.74% | **74.86%** | 是 |
| 10 | PCEN | last_block | 2 | 72.13% | **61.67%** | **117.04%** | 72.13% | 否 |
| 10 | PCEN | all | 2 | 72.13% | 60.76% | 115.32% | 72.13% | 否 |
| 20 | MFCC | head | 18 | **73.01%** | 42.96% | 86.59% | **63.22%** | 是 |
| 20 | MFCC | last_block | 11 | 72.08% | 42.36% | 85.38% | 61.54% | 否 |
| 20 | MFCC | all | 3 | 61.18% | **47.84%** | **96.42%** | 58.99% | 否 |
| 20 | LogMel | head | 10 | **78.20%** | 58.79% | 92.50% | 72.33% | 否 |
| 20 | LogMel | last_block | 2 | 76.71% | 60.33% | 94.92% | 72.81% | 否 |
| 20 | LogMel | all | 1 | 77.24% | **62.71%** | **98.67%** | **76.22%** | 是 |
| 20 | PCEN | head | 6 | 76.69% | 59.19% | 112.34% | 76.69% | 否 |
| 20 | PCEN | last_block | 10 | **77.75%** | **61.04%** | **115.85%** | **77.75%** | 是 |
| 20 | PCEN | all | 15 | 77.54% | 60.51% | 114.85% | 77.54% | 否 |

#### 共同DB3V held-out上的5/10/20-shot结果

| 特征 | Shot/策略 | Accuracy | Balanced Accuracy | Macro-F1 | 相对零样本变化 | Top-3 |
|---|---|---:|---:|---:|---:|---:|
| MFCC | 0/baseline | 57.05% | 51.91% | 50.17% | — | 84.69% |
| MFCC | 5/head | 60.97% | 57.57% | 55.58% | +5.42 pp | 86.50% |
| MFCC | 10/head | 62.82% | 59.87% | 56.94% | +6.77 pp | 87.01% |
| MFCC | 20/head | 67.16% | 64.16% | 61.38% | **+11.22 pp** | 88.53% |
| LogMel | 0/baseline | 70.71% | 70.49% | 67.34% | — | 90.99% |
| LogMel | 5/head | 72.19% | 71.04% | 67.64% | +0.31 pp | **92.00%** |
| LogMel | 10/last_block | **73.05%** | 72.38% | 68.59% | +1.25 pp | 91.97% |
| LogMel | 20/all | 72.93% | **72.62%** | 68.86% | **+1.53 pp** | 91.67% |
| PCEN | 0/baseline | 63.46% | 63.04% | 59.52% | — | 85.79% |
| PCEN | 5/all | 71.09% | 69.39% | 65.82% | +6.30 pp | 90.82% |
| PCEN | 10/head | 70.57% | 69.68% | 65.95% | +6.44 pp | 89.14% |
| PCEN | 20/last_block | **75.80%** | **73.66%** | **70.60%** | **+11.09 pp** | **92.42%** |

三个特征的DB3V Macro-F1均随support规模总体上升。10-shot相对5-shot的边际
收益在LogMel和PCEN上很小，而20-shot对MFCC和PCEN带来明显跃升。所有链路中，
`PCEN + 20-shot + last_block` 的DB3V绝对Macro-F1最高，为70.60%；但它是否适合作为
通用模型还必须结合Xeno和BirdSet复测。

| 特征 | Xeno Macro-F1（0→5→10→20） | Xeno保留率（5→10→20） | BirdSet Top-1（0→5→10→20） | BirdSet纯单物种Macro-F1（0→5→10→20） |
|---|---:|---:|---:|---:|
| MFCC | 49.62% → 46.10% → 47.87% → 40.47% | 92.90% → 96.49% → 81.57% | 17.32% → 10.72% → 11.93% → 24.17% | 9.77% → 7.00% → 7.48% → 14.92% |
| LogMel | 63.56% → 60.05% → 62.13% → 63.45% | 94.48% → 97.75% → **99.83%** | 18.27% → 18.81% → 18.86% → **19.47%** | 10.12% → 9.19% → 9.90% → **10.46%** |
| PCEN | 52.69% → 59.92% → 60.74% → **61.51%** | 113.72% → 115.29% → **116.73%** | 19.57% → 9.81% → 12.70% → 9.70% | 8.08% → 9.42% → 9.86% → 9.16% |

在原27组单种子、BatchNorm冻结策略的历史实验中，综合链路为
`Xeno-canto → LogMel → DS-CNN → DB3V 20-shot all微调（BatchNorm冻结）`：
共同DB3V Macro-F1为68.86%，比零样本提高1.53个百分点，同时保留99.83%的
Xeno Macro-F1，并提高BirdSet Top-1和纯单物种Macro-F1。若部署目标只强调DB3V
三个地区，则 `PCEN → 20-shot → last_block` 更优，Macro-F1达到70.60%；但其
BirdSet Top-1从19.57%降至9.70%，地区适配的跨域代价明显。MFCC 20-shot虽在
BirdSet上意外改善，但Xeno Macro-F1只保留81.57%，不宜替换当前通用基准。

结果与实现来源：

- 微调实现：`src/experiments/fine_tune_db3v.py`
- 固定划分：`src/dataset_processing/output/{MFCC,LogMel,PCEN}_DB3V_external_split_8class/` 及对应的 `_10shot_8class/`、`_20shot_8class/`
- 27组策略统一选型记录：`src/experiments/DB3V_fewshot_comparison_8class/selection_summary.csv`
- 共同测试集最终对比：`src/experiments/DB3V_fewshot_comparison_8class/comparison_summary.csv`
- 5-shot原始记录：`src/experiments/DB3V_fewshot_8class/`
- 10/20-shot原始记录：`src/experiments/DB3V_fewshot_10shot_8class/`、`src/experiments/DB3V_fewshot_20shot_8class/`
- 共同零样本报告：`src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DB3V_common_20shot_heldout_evaluation.json`
- 每个候选目录中的 `DS_CNN_Model.fewshot.json` 和 history 保存选型、Xeno复测及逐轮记录；选中策略目录另含共同DB3V与BirdSet报告。

复现单组策略时使用以下命令；将三个目录和参数替换为同一特征及上表对应策略即可：

```powershell
& .\.venv\Scripts\python.exe src\experiments\fine_tune_db3v.py `
  --base-model-dir src\experiments\Feature_comparison_8class\LogMel `
  --support-dir src\dataset_processing\output\LogMel_DB3V_external_split_8class `
  --xeno-dataset-dir src\dataset_processing\output\LogMel_dataset_A_8class `
  --output-dir src\experiments\DB3V_fewshot_8class\LogMel\head `
  --policy head `
  --learning-rate 0.0003 `
  --epochs 30 `
  --patience 6 `
  --batch-size 32 `
  --seed 42
```

复现10-shot时把划分与输出目录改为对应的 `10shot` 目录，并加入
`--validation-recordings-per-stratum 2`；复现20-shot时改为 `20shot` 目录并使用
`--validation-recordings-per-stratum 4`。跨shot比较必须把评估数据目录统一指向
匹配特征的 `_DB3V_external_split_20shot_8class`，否则测试录音集合不同。

选型完成后，分别使用 `evaluate_db3v.py` 对匹配特征的 held-out 目录评估，并使用
`evaluate_birdset_ssw.py` 对匹配特征的完整 BirdSet SSW 目录复测。每个策略目录
保留 `DS_CNN_Model.fewshot.json`、逐轮 history 和模型权重；每个shot选中的三个
策略目录另外保留DB3V与BirdSet报告。CSV汇总只做索引，原始JSON是最终可追溯记录。

### DB3V严格微调策略与多随机种子补充实验

为补齐旧实验中“`head`实际训练两个Dense层”“`all`仍冻结BatchNorm”以及只有单一
随机种子的问题，新增四个定义无歧义的严格策略。模型总参数为30,056：

| 策略 | 可训练范围 | 可训练参数 | BatchNorm可训练 | Xeno replay |
|---|---|---:|---|---|
| `head_only` | 仅最终8类softmax Dense层 | 1,032 | 否 | 否 |
| `bn_head` | 全部BatchNorm与最终softmax层 | 1,480 | 是 | 否 |
| `bn_head_replay` | 与`bn_head`相同 | 1,480 | 是 | 是，Xeno训练切片与DB3V support切片1:1 |
| `full` | 全部层，包括BatchNorm | 29,608 | 是 | 否 |

实验固定使用原来的嵌套DB3V support（5/10/20-shot实际为120/239/461条录音）和
10,197条共同held-out。随机种子42、123、2026分别改变support内部训练/验证划分、
Replay抽样、批次顺序和TensorFlow随机操作；外部support身份和最终held-out身份保持
固定，因此本表衡量训练与内部选型随机性，不包含重新抽取外部support带来的方差。
Replay只从Xeno-canto基准训练划分按类别均衡抽样，不读取Xeno验证、DB3V held-out或
BirdSet。每个结果均为3个随机种子的均值±样本标准差（`ddof=1`）。

每个“特征×shot”先按三个seed的平均适配分数选策略；适配分数仍为support内部验证
录音级Macro-F1乘以截断到1的Xeno Macro-F1保留率，DB3V held-out不参与选型：

| 特征 | shot | 选中严格策略 | Xeno Macro-F1 | Xeno变化 | DB3V共同held-out Macro-F1 |
|---|---:|---|---:|---:|---:|
| MFCC | 5 | `head_only` | 47.79% ± 0.60% | -1.82 ± 0.60 pp | 53.94% ± 0.46% |
| MFCC | 10 | `head_only` | 48.43% ± 1.35% | -1.19 ± 1.35 pp | 53.74% ± 2.07% |
| MFCC | 20 | `bn_head_replay` | 49.64% ± 0.35% | +0.03 ± 0.35 pp | 56.64% ± 1.87% |
| LogMel | 5 | `head_only` | 62.33% ± 0.76% | -1.23 ± 0.76 pp | 67.72% ± 0.12% |
| LogMel | 10 | `head_only` | 61.20% ± 0.81% | -2.36 ± 0.81 pp | 67.84% ± 0.05% |
| LogMel | 20 | `bn_head_replay` | 60.95% ± 0.49% | -2.61 ± 0.49 pp | 66.97% ± 0.25% |
| PCEN | 5 | `bn_head_replay` | 62.58% ± 3.37% | +9.89 ± 3.37 pp | 67.64% ± 2.64% |
| PCEN | 10 | `head_only` | 58.01% ± 2.65% | +5.32 ± 2.65 pp | 65.27% ± 3.39% |
| **PCEN** | **20** | **`full`** | **63.66% ± 0.44%** | **+10.98 ± 0.44 pp** | **70.50% ± 0.40%** |

当前正式零样本基准的LogMel链路上，四个严格策略的完整结果如下；单元格为
“DB3V Macro-F1 / Xeno Macro-F1变化”，均以百分点表示：

| shot | `head_only` | `bn_head` | `bn_head_replay` | `full` |
|---:|---:|---:|---:|---:|
| 5 | **67.72±0.12 / -1.23±0.76** | 67.28±0.15 / -1.27±0.74 | 67.38±0.08 / -1.11±0.30 | 67.34±0.17 / **-0.89±0.05** |
| 10 | **67.84±0.05 / -2.36±0.81** | 66.57±0.33 / -3.19±0.45 | 67.34±0.40 / **-1.59±0.63** | 67.00±0.38 / -3.73±1.43 |
| 20 | **67.99±0.25 / -1.83±0.55** | 66.62±0.32 / -3.38±0.69 | 66.97±0.25 / -2.61±0.49 | 67.53±1.09 / -4.39±1.29 |

严格实验的主要结论是：

- `PCEN + 20-shot + full` 是按预先定义适配分数选中且DB3V结果最高的严格策略，
  DB3V Macro-F1为70.50%±0.40%，同时Xeno Macro-F1相对基模型提高
  10.98±0.44个百分点。
- `PCEN + 20-shot + bn_head_replay` 的DB3V Macro-F1为69.32%±0.45%，但Xeno提升
  更大（+12.63±0.13个百分点），说明Replay确实改善基准域保留；其平均适配分数
  略低于`full`，因此未被选中。
- LogMel对严格全量微调较敏感；在共同held-out上，三个shot均由`head_only`取得
  最高DB3V均值。20-shot `full`的标准差增至1.09个百分点且Xeno下降4.39个百分点。
- 这些新策略尚未在BirdSet多标签声景上复测，所以`PCEN + 20-shot + full`当前只
  是DB3V地区适配候选，不能替换唯一的Xeno-canto LogMel零样本基准或宣称为通用模型。

完整36组聚合结果和108个逐seed结果是本节数值的直接来源：

- 训练与策略定义：`src/experiments/fine_tune_db3v.py`
- 批量执行与统计实现：`src/experiments/run_db3v_ablation_multiseed.py`
- 实验协议：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/experiment_protocol.json`
- 36组均值/标准差：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/aggregate.csv`
- 108个逐seed索引：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/runs.csv`
- 原始模型、逐轮历史、Xeno遗忘和DB3V报告：
  `src/experiments/DB3V_fewshot_ablation_multiseed_8class/{MFCC,LogMel,PCEN}/{5shot,10shot,20shot}/{head_only,bn_head,bn_head_replay,full}/seed_<seed>/`
- DB3V数据来源与许可：[Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734)；
  固定划分协议位于被Git排除的
  `src/dataset_processing/output/<feature>_DB3V_external_split*_8class/split_manifest.json`。

从项目根目录可复现或重建汇总：

```powershell
& .\.venv\Scripts\python.exe src\experiments\run_db3v_ablation_multiseed.py
& .\.venv\Scripts\python.exe src\experiments\run_db3v_ablation_multiseed.py `
  --summarize-only
```

### BirdSet grouped 5-shot 多标签适配结果

已对 Xeno-canto 训练的 MFCC、LogMel、PCEN 三条 DS-CNN 链路完成 BirdSet SSW
小样本适配。划分固定随机种子42，并以原始长录音为隔离单位：support包含10条长
录音、684个五秒片段和3,420个一秒窗口；held-out包含另外201条长录音、18,994个
片段和94,970个窗口。三种特征使用完全相同的片段身份，两个集合没有原始录音重叠。

这里的“5-shot”是按“每类至少5个正片段”构建 grouped support 的目标，不是最终
每类恰好5个独立样本。保持长录音完整后，各类support正片段数为
`[193, 92, 21, 179, 10, 2, 252, 37]`；其中 `Setophaga_ruticilla` 在完整目标
子集中也只有同一长录音里的2个片段，无法达到5-shot。

#### 实验协议

| 项目 | 固定配置 |
|---|---|
| 执行时间 | 2026-07-26（Asia/Tokyo；原始 JSON 使用 UTC 时间戳） |
| 基准权重 | `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.h5` |
| 任务变换 | 从八类softmax切换为八类sigmoid；multi-hot标签；binary cross-entropy |
| support内部切分 | 按长录音固定拆为7条训练录音/554个片段与3条验证录音/130个片段 |
| 验证覆盖 | 验证集覆盖6类；两个极稀有类必须留在训练集，才能保证训练集覆盖全部8类 |
| 类别平衡 | 按support片段正类频次倒数生成权重，再分配到每个片段的5个一秒窗口 |
| 随机性 | TensorFlow/NumPy随机种子均为42 |
| 优化器与批量 | Adam，batch size 32 |
| 最大轮次与早停 | 最多30轮；选型分数连续6轮不改善即停止并恢复最佳权重 |
| 微调策略 | `head`：分类头，LR 3e-4；`last_block`：末端可分离卷积块和分类头，LR 1e-4；`all`：除BatchNorm外全网络，LR 3e-5 |
| 选型指标 | support验证片段Top-1任意目标命中率 × 截断到1的Xeno Macro-F1保留率 |
| 最终训练 | 重新加载原始Xeno权重，按选定轮次使用全部10条support长录音训练 |
| 泄漏控制 | `fine_tune_birdset.py`不接受held-out参数；全部选型完成后才读取held-out |
| 运行环境 | Python 3.11.9、TensorFlow 2.19.0、NumPy 2.1.3 |

验证集只覆盖6类且未做阈值校准，因此阈值0.5下的多标签Macro-F1不作为选型依据；
主选型指标使用不依赖阈值的Top-1任意目标命中率，并同时约束Xeno保留率。

#### 九组候选策略

| 特征 | 策略 | LR | 最佳轮次 | support验证Top-1 | 选型Xeno Macro-F1 | Xeno保留率 | 选型分数 | 采用 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MFCC | head | 3e-4 | 2 | 13.08% | **43.67%** | **88.01%** | 11.51% | 否 |
| MFCC | last_block | 1e-4 | 4 | 18.46% | 41.97% | 84.58% | 15.61% | 否 |
| MFCC | all | 3e-5 | 13 | **23.08%** | 41.61% | 83.87% | **19.35%** | 是 |
| LogMel | head | 3e-4 | 1 | **16.15%** | **53.78%** | **84.62%** | **13.67%** | 是 |
| LogMel | last_block | 1e-4 | 1 | 9.23% | 52.47% | 82.56% | 7.62% | 否 |
| LogMel | all | 3e-5 | 2 | 10.77% | 52.77% | 83.03% | 8.94% | 否 |
| PCEN | head | 3e-4 | 3 | 19.23% | **51.46%** | **97.67%** | **18.78%** | 是 |
| PCEN | last_block | 1e-4 | 1 | 15.38% | 43.37% | 82.31% | 12.66% | 否 |
| PCEN | all | 3e-5 | 1 | **20.00%** | 45.42% | 86.20% | 17.24% | 否 |

#### 独立held-out与跨域复测

BirdSet结果在未参与训练、选型和早停的201条长录音上计算；Top-1/Top-3以五秒
片段为粒度，对同一片段的五个一秒窗口概率取均值。纯单物种Macro-F1仅统计全局
singleton片段中在held-out有支持的7类。Xeno和DB3V仍以完整录音聚合概率后的
八类argmax单标签Macro-F1计算，所以不依赖softmax概率和为1；DB3V没有参与本轮
训练，因此使用完整10,658条录音复测。

| 特征/选中策略 | BirdSet Top-1（前→后） | BirdSet Top-3（前→后） | BirdSet纯单物种Macro-F1（前→后） | Xeno Macro-F1（前→后） | DB3V Macro-F1（前→后） |
|---|---:|---:|---:|---:|---:|
| MFCC/all，13轮 | 17.01% → 24.23% | 44.49% → 53.67% | 9.72% → 17.75% | 49.62% → 42.43% | 50.47% → 41.58% |
| LogMel/head，1轮 | 17.87% → 21.09% | 41.29% → 52.11% | 9.95% → 15.15% | 63.56% → 55.80% | 67.62% → 57.94% |
| **PCEN/head，3轮** | **19.37% → 35.70%** | **25.96% → 64.64%** | **8.04% → 24.66%** | **52.69% → 59.44%** | **59.93% → 66.66%** |

PCEN分类头微调是本轮唯一在三个数据域上同时取得正增益的链路：BirdSet Top-1
提高16.33个百分点、纯单物种Macro-F1提高16.62个百分点，Xeno和DB3V
Macro-F1分别提高6.75和6.73个百分点。因此，面向BirdSet真实多鸟声景的推荐
增强链路是 `Xeno-canto → PCEN → DS-CNN → BirdSet grouped 5-shot head微调`。
MFCC和LogMel虽然也改善BirdSet，但Xeno与DB3V Macro-F1分别下降约7–10个百分点，
不宜替换通用基准。

正式的从零训练基准仍是 `Xeno-canto → LogMel → DS-CNN`；上述PCEN模型是
BirdSet support适配模型，不能与基准训练集选择混为一谈。本轮只运行一个随机种子，
grouped support高度不均衡，一秒窗口使用五秒弱标签，内部验证缺少两个稀有类，
也尚未估计置信区间，因此当前增益仍需多种子重复实验确认。

结果与实现来源：

- 微调实现：`src/experiments/fine_tune_birdset.py`
- 固化实验协议与官方来源：`src/experiments/BirdSet_fewshot_8class/experiment_protocol.json`
- 固定划分：`src/dataset_processing/output/{MFCC,LogMel,PCEN}_BirdSet_external_split_8class/split_manifest.json`
- 九组策略统一选型记录：`src/experiments/BirdSet_fewshot_8class/selection_summary.csv`
- 三条选中链路跨域对比：`src/experiments/BirdSet_fewshot_8class/comparison_summary.csv`
- 每组原始选型、逐轮历史和最终Xeno复测：`src/experiments/BirdSet_fewshot_8class/{MFCC,LogMel,PCEN}/{head,last_block,all}/`
- BirdSet零样本原始报告：`src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/BirdSet_SSW_heldout_evaluation.json`
- 选中模型的BirdSet与DB3V原始报告：对应策略目录中的 `BirdSet_SSW_heldout_evaluation.json` 与 `DB3V_full_evaluation.json`
- CSV重建脚本：`src/experiments/summarize_birdset_fewshot.py`

复现推荐的PCEN单组策略：

```powershell
& .\.venv\Scripts\python.exe src\experiments\fine_tune_birdset.py `
  --base-model-dir src\experiments\Feature_comparison_8class\PCEN `
  --support-dir src\dataset_processing\output\PCEN_BirdSet_external_split_8class\support `
  --xeno-dataset-dir src\dataset_processing\output\PCEN_dataset_A_8class `
  --output-dir src\experiments\BirdSet_fewshot_8class\PCEN\head `
  --policy head `
  --learning-rate 0.0003 `
  --epochs 30 `
  --patience 6 `
  --batch-size 32 `
  --seed 42
```

三种特征的三个策略全部完成并生成held-out报告后，可运行
`& .\.venv\Scripts\python.exe src\experiments\summarize_birdset_fewshot.py`
从原始JSON重建两张CSV。

## 严格INT8量化接口与精度测试

已完成全部15条FP32选中链路的训练后静态量化（PTQ）和同口径全量复测：

- 3条零样本基准：MFCC、LogMel、PCEN；
- 9条DB3V小样本链路：三种特征各自的5/10/20-shot选中策略；
- 3条BirdSet grouped 5-shot链路：三种特征各自的选中策略。

量化结果不用于重新选择FP32策略。代表性数据固定为256个训练窗口：零样本只使用
Xeno-canto训练特征；小样本链路在Xeno训练特征之外只加入对应DB3V或BirdSet
support。任何held-out和Xeno验证数据都不参与校准。

### 输入接口与严格性

| 特征 | 特征提取输出 | TFLite输入 | 说明 |
|---|---|---|---|
| MFCC | `float32 [batch,32,13]` | `int8 [batch,32,13,1]` | 13维MFCC |
| LogMel | `float32 [batch,32,40]` | `int8 [batch,32,40,1]` | 40维Log-Mel |
| PCEN | `float32 [batch,32,40]` | `int8 [batch,32,40,1]` | 形状同LogMel，但语义和校准数据独立 |

每条链路从TFLite模型读取自己的input scale和zero-point，按
`q=clip(round(x/scale)+zero_point,-128,127)` 量化输入，再使用输出scale和
zero-point反量化后执行原有时间聚合。15个模型全部满足：

- 仅允许 `TFLITE_BUILTINS_INT8`；
- 输入、权重、激活和输出走整数路径，输入/输出均为int8；
- 浮点张量数为0，图内没有 `QUANTIZE`/`DEQUANTIZE`；
- int32偏置和乘加累积属于标准整数内核；
- 单模型大小为49,968–49,984 bytes。

### 零样本INT8结果

BirdSet使用与support隔离的201条长录音；DB3V使用完整10,658条录音。

| 特征 | Xeno Macro-F1（FP32→INT8） | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC | 49.62% → 48.77% (-0.85 pp) | 17.01% → 15.43% (-1.57 pp) | 9.72% → 9.41% (-0.31 pp) | 50.47% → 51.25% (+0.78 pp) |
| **LogMel** | **63.56% → 59.01% (-4.55 pp)** | **17.87% → 18.48% (+0.61 pp)** | **9.95% → 10.32% (+0.37 pp)** | **67.62% → 65.94% (-1.69 pp)** |
| PCEN | 52.69% → 35.07% (-17.62 pp) | 19.37% → 21.55% (+2.17 pp) | 8.04% → 9.28% (+1.24 pp) | 59.93% → 50.24% (-9.69 pp) |

MFCC量化最稳定，但LogMel量化后的Xeno和DB3V绝对Macro-F1仍最高，因此严格INT8
零样本基准推荐保持 `Xeno-canto → LogMel → DS-CNN`。PCEN不能直接沿用FP32结论。

### DB3V小样本INT8结果

DB3V统一使用20-shot support之外的10,197条共同held-out；BirdSet使用完整目标
子集，因为DB3V适配没有读取BirdSet。

| 链路 | Xeno Macro-F1（FP32→INT8） | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
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

FP32的DB3V最高链路是PCEN 20-shot，但严格INT8后Macro-F1从70.60%降至54.46%。
严格INT8部署推荐改为
`Xeno-canto → LogMel → DS-CNN → DB3V 10-shot last_block`：共同held-out
Macro-F1为69.70%，Xeno Macro-F1为58.54%，且BirdSet两项指标均提高。

### BirdSet小样本INT8结果

BirdSet使用与support按长录音隔离的201条held-out；DB3V完整集没有参与本轮训练。

| 链路 | Xeno Macro-F1（FP32→INT8） | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC/5-shot/all | 42.43% → 39.62% (-2.81 pp) | 24.23% → 24.97% (+0.74 pp) | 17.75% → 18.72% (+0.97 pp) | 41.58% → 41.81% (+0.23 pp) |
| **LogMel/5-shot/head** | **55.80% → 53.68% (-2.12 pp)** | **21.09% → 23.82% (+2.73 pp)** | **15.15% → 16.97% (+1.82 pp)** | **57.94% → 58.08% (+0.14 pp)** |
| PCEN/5-shot/head | 59.44% → 34.35% (-25.09 pp) | 35.70% → 37.37% (+1.67 pp) | 24.66% → 17.47% (-7.19 pp) | 66.66% → 45.97% (-20.69 pp) |

PCEN INT8的BirdSet Top-1仍最高，但纯单物种Macro-F1下降7.19个百分点，且Xeno和
DB3V分别下降25.09和20.69个百分点。该Top-1上升不能解释为稳定增益。兼顾真实
多标签声景、纯单物种和跨域保留时，严格INT8推荐
`Xeno-canto → LogMel → DS-CNN → BirdSet grouped 5-shot head`。

### 量化结论、来源与复现

- MFCC是最耐PTQ量化的特征；小样本链路的主要Macro-F1变化均在约3个百分点内。
- LogMel是严格INT8下绝对精度与稳定性的较优折中。
- PCEN各数据集输入饱和率均小于0.001%，但Xeno/DB3V Macro-F1仍普遍下降
  约10–27个百分点，说明输入截断不是主因，更可能是内部激活量化敏感，仍需逐层
  量化误差分析确认。
- 如果必须部署PCEN，需要进行量化感知训练后重新执行全部held-out评估。

结果与实现来源：

- 输入、转换与推理接口：`src/experiments/int8_inference.py`
- 全链路评估入口：`src/experiments/evaluate_int8_experiments.py`
- 详细实验说明：`src/experiments/INT8_quantization_8class/README.md`
- 严格性、校准隔离和链路清单：`src/experiments/INT8_quantization_8class/experiment_protocol.json`
- 每条链路的scale/zero-point：`src/experiments/INT8_quantization_8class/input_interfaces.json`
- 15条统一结果：`src/experiments/INT8_quantization_8class/summary.csv`
- 模型及逐类/逐粒度原始报告：`src/experiments/INT8_quantization_8class/models/<chain_id>/`

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --representative-samples 256 `
  --batch-size 128 `
  --num-threads 4
```

## 指标与粒度解释

- **Accuracy**：整体预测正确比例，受类别不平衡影响。
- **Balanced Accuracy**：各类别召回率的算术平均，更适合类别不均衡数据。
- **Macro-F1**：先分别计算每类 F1 再等权平均，是当前单标签八类任务的主要综合指标。
- **Top-3**：真实类别是否位于模型概率最高的三个类别中。
- **一秒切片级**：衡量瞬时识别能力，容易受静音、弱鸣声和局部噪声影响。
- **录音/片段级**：平均同一来源连续窗口的 softmax 概率，衡量时间聚合后的识别能力。
- **BirdSet任意目标命中率**：用于 BirdSet 内部多标签验证，表示概率最高类别是否命中任一真实标签；其定义不同于单标签 Accuracy。

原27组单seed结果表明：增加DB3V support总体能够提高三个地区的识别能力，但不同
特征的跨域代价差异很大。LogMel 20-shot在DB3V、Xeno和BirdSet之间取得较稳健的
平衡；旧PCEN 20-shot链路取得最高DB3V Macro-F1，但BirdSet Top-1明显下降。新增
三seed严格实验中，PCEN 20-shot Full Fine-Tuning达到70.50%±0.40%的DB3V
Macro-F1并提高Xeno指标，但尚未复测BirdSet，所以只能视为地区适配候选。反向使用
BirdSet support适配时，PCEN分类头微调在BirdSet、Xeno和DB3V上都取得正增益，说明
“地区单标签适配”和“跨数据集多标签声景适配”必须保留为两条独立实验线路。
这些是FP32结论；新增严格策略尚未量化。已有严格INT8实验中PCEN发生严重退化，
部署排序必须重新按量化报告解释。当前已验证的INT8推荐仍使用LogMel：零样本使用
原始LogMel DS-CNN，DB3V使用10-shot `last_block`，BirdSet使用grouped 5-shot
`head`。
