# 鸟类鸣声分类项目

本项目仅使用 Xeno-canto 鸟鸣录音训练基准分类器。BirdSet SSW 与 DB3V
分别作为跨数据集声景泛化和特定地区泛化的独立测试集，并各自提供与测试录音
隔离的小样本 support 集。当前推荐模型是八类 Xeno-canto Log-Mel DS-CNN；
两个外部 held-out test 均不参与训练、早停或模型选择。BirdSet grouped
5/10/20-shot 多标签适配和 DB3V 5/10/20-shot 单标签适配均已完成。

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
        ├── fine_tune_birdset.py             # BirdSet grouped 5/10/20-shot多标签适配
        ├── fine_tune_db3v.py                # DB3V 5/10/20-shot地区适配
        ├── run_birdset_ablation_multiseed.py # BirdSet严格策略多随机种子实验与汇总
        ├── run_db3v_ablation_multiseed.py   # DB3V严格策略多随机种子实验与汇总
        ├── int8_inference.py                # 特征感知严格INT8接口与批量推理
        ├── evaluate_int8_experiments.py     # 零样本/小样本INT8统一复测
        ├── evaluate_db3v.py                 # DB3V区域评估
        ├── evaluate_birdset_ssw.py          # BirdSet多标签评估
        ├── audit_db3v_source_split.py       # DB3V源录音级泄漏与嵌套审计
        ├── build_research_evidence_matrix.py # 研究证据完整性清单
        ├── analyze_board_power.py           # 外部功耗仪采样积分
        ├── convert.py                       # 模型转换入口
        ├── construct_model/                 # 模型结构
        ├── TinyML_model_8class/             # MFCC兼容基准模型与报告
        ├── Feature_comparison_8class/       # Xeno三特征正式实验
        ├── BirdSet_fewshot_ablation_multiseed_8class/ # BirdSet严格108组实验
        ├── BirdSet_strict_INT8_quantization_8class/ # BirdSet严格链路INT8复测
        ├── DB3V_fewshot_ablation_multiseed_8class/ # 108组严格策略实验与统计
        ├── DB3V_strict_INT8_quantization_8class/ # DB3V严格链路INT8与BirdSet复测
        ├── ZeroShot_strict_INT8_quantization_8class/ # 当前零样本严格INT8
        ├── Board_replay_8class/             # 板端复刻协议与部署证据
        ├── legacy_single_seed_reference.md  # 已淘汰单种子链路必要结论
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
- 当前 MFCC 特征形状为 `(32, 13)`：Xeno-canto 训练/验证数组分别为 `(14545, 32, 13)` 和 `(3652, 32, 13)`；DB3V 数组为 `(85264, 32, 13)`，对应10,658个八秒块和1,363条独立原始来源录音。
- Log-Mel 和 PCEN 的形状均为 `(32, 40)`，录音划分、切片索引及样本数与 MFCC 完全一致，因此特征对比不受数据划分差异影响。
- DB3V 保持为独立地区测试集。评估同时报告一秒切片、八秒块和原始来源录音三级指标；当前主指标先聚合同一原始来源的全部八秒块，使每条独立来源只投一票。
- 模型、`*.labels.json` 标签侧车、训练历史、验证摘要与 DB3V JSON/CSV 报告会保存在同一个模型目录中。转换为 TFLite 或 C 头文件时，单独运行 `convert.py`。
- Xeno-canto API 密钥不写入仓库；运行爬取脚本前必须设置
  `XENO_CANTO_API_KEY` 环境变量。音频标准化依赖 FFmpeg。

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

截至 **2026-09-01**，当前标准实验已经完成从Xeno-canto FP32基准、两个外部域零样本评估、5/10/20-shot严格适配，到57个严格INT8模型的同口径复测。DB3V已经按原始Xeno-canto来源ID重新分组，旧的八秒块主指标与旧support规模不再用于当前结论。

### 研究证据核对

| 研究问题 | 当前状态 | 已有证据或缺口 |
|---|---|---|
| 跨数据集泛化 | 完成 | Xeno内部验证、DB3V零适应、BirdSet零适应；三种特征使用固定模型与固定test |
| 少样本适应 | 完成 | DB3V与BirdSet均完成3特征×3 shot×4策略×3 seed＝108个FP32模型；每个运行保存Xeno遗忘 |
| INT8稳定性 | 完成 | 3个零样本、27个DB3V适配、27个BirdSet适配模型均完成同模型同test的FP32–INT8配对 |
| Flash与RAM | 完成 | 基于保留ELF和链接脚本记录三个零样本固件的静态Flash/RAM与tensor arena |
| 板端输出一致性、延迟 | 历史完成，当前模型包待刷新 | 历史57模型有570组TFLite–TFLM输出比较、0不一致、最大0 LSB，并有延迟；其commit不等于本轮模型包 |
| 板端特征一致性 | **受阻** | 当前MCU可运行，但`/dev/ttyACM0`发送命令超时；修复VCP/LPUART RX链路后才能抓取同PCM特征 |
| 功耗 | **受阻** | 尚无电流探头或功耗仪；ST-Link目标电压不能替代功耗测量。分析脚本与区间模板已准备 |

机器可读核对表由`src/experiments/build_research_evidence_matrix.py`生成到
`src/experiments/research_evidence_matrix.json`。板端未完成项不会用桌面模拟值冒充。

## 当前正式实验角色

| 数据集 | 角色 | 是否允许参与基准选型 |
|---|---|---|
| Xeno-canto | 唯一基准训练与内部验证集 | 是；只使用1,920条训练和480条验证录音 |
| DB3V | 特定地区单标签独立测试；隔离support用于地区适配 | 否；test不参与梯度、早停、策略或epoch选择 |
| BirdSet SSW | 跨数据集多标签声景独立测试；隔离support用于声景适配 | 否；test不参与梯度、早停、策略或epoch选择 |

所有正式链路都从相同Xeno-canto DS-CNN权重出发：

```text
Xeno-canto训练
  ├── MFCC / LogMel / PCEN FP32基准
  ├── DB3V与BirdSet零样本泛化
  ├── DB3V 5/10/20-shot × 4策略 × 3 seed
  ├── BirdSet 5/10/20-shot × 4策略 × 3 seed
  └── 预先选中的FP32模型 → 严格INT8 → 三域同口径复测
```

### 为什么BirdSet和DB3V不作为正式基准

这两个数据集并非技术上不能训练，而是不能在保留当前研究问题的同时充当基准训练集。

| 候选来源 | 已完成的反事实训练结果 | 不能作为正式基准的原因 |
|---|---|---|
| BirdSet SSW | LogMel验证Top-1任意目标46.59%；跨到DB3V的历史八秒块级Macro-F1仅9.97% | 当前使用官方多标签`test_5s`声景；211条长录音且类别极不均衡，五秒弱标签复制到一秒窗口会引入噪声；训练后会消耗唯一跨数据集test |
| DB3V | LogMel同地区留出Macro-F1 90.33%；跨BirdSet Top-1 19.79% | 训练与验证共享三个地区的设备和背景，只能证明同地区识别；训练后无法继续回答未适应地区泛化 |
| Xeno-canto | LogMel内部验证Macro-F1 63.56%，零样本DB3V源录音级72.87%，BirdSet Top-1 17.78% | 2,400条均衡焦点录音可作无会话泄漏内部选型，同时保留两个外部域 |

反事实实验的指标任务和粒度不同，只用于验证数据角色，不能横向排名。来源见
`src/experiments/BirdSet_feature_comparison_8class/`和
`src/experiments/DB3V_baseline_diagnostic_8class/`。

## 数据划分与指标粒度

### Xeno-canto

2,400条焦点录音按完整`session_key`固定划分：每类240条训练、60条验证，总计1,920/480。训练与验证不共享录音或会话；每条录音最多8个一秒窗口，三种特征使用相同录音和窗口身份。

### DB3V源录音级划分

DB3V文件名中的原始Xeno-canto来源ID是最小隔离单位。同一来源产生的多个八秒块必须整体进入support或test。

| shot | support源录音 | support八秒块 | 源录音短缺 | 公共test源录音 | 公共test八秒块 | 公共test一秒切片 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 116 | 1,009 | 4 | 970 | 7,660 | 61,280 |
| 10 | 219 | 1,665 | 21 | 970 | 7,660 | 61,280 |
| 20 | 393 | 2,998 | 87 | 970 | 7,660 | 61,280 |

5/10/20-shot support源录音集合严格嵌套，三种特征身份一致。审计结果：

- support/test原始来源重叠为0；
- DB3V与Xeno训练1,920条、验证480条的来源重叠均为0；
- 数组、manifest、源录音和八秒块数量一致；
- 审计状态为`passed: true`。

完整审计见
`src/experiments/DB3V_fewshot_ablation_multiseed_8class/split_audit.json`。

### BirdSet grouped划分

shot表示每类期望正五秒片段数，但隔离单位是完整长录音。

| shot | support长录音 | support五秒片段 | 对应test长录音 | 对应test五秒片段 |
|---:|---:|---:|---:|---:|
| 5 | 10 | 684 | 201 | 18,994 |
| 10 | 11 | 699 | 200 | 18,979 |
| 20 | 14 | 1,413 | 197 | 18,265 |

横向比较统一使用最大20-shot support之外的197条长录音、18,265个五秒片段和91,325个一秒窗口。三种特征的样本身份SHA-256一致。`Setophaga_ruticilla`只有同一长录音中的2个正片段，因此明确记录shot缺口，不复制样本补齐。

### 当前主要指标

| 数据集 | 主指标与粒度 |
|---|---|
| Xeno-canto | 480条内部验证录音级Macro-F1；先平均同一录音的一秒窗口概率 |
| DB3V | 原始来源录音级Macro-F1；先聚合八秒块内窗口，再聚合同一来源全部八秒块 |
| BirdSet | 五秒片段Top-1命中任一真实标签；另报全局纯单物种片段的支持类Macro-F1 |

DB3V的一秒切片和八秒块指标仍保存在报告中，但不再作为主结论。源录音级数值通常高于旧八秒块级数值，是因为长来源不再按块数重复加权，而不是模型本身发生变化。

## Xeno-canto零样本基准与严格INT8

以下DB3V结果使用完整1,363条原始来源录音；BirdSet使用统一公共test。每格均为同一模型、同一test的`FP32 → INT8`。

| 特征 | Xeno Macro-F1 | DB3V源录音Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC | 49.62% → 48.77% | 55.55% → 56.75% | 16.99% → 15.36% | 9.74% → 9.40% |
| **LogMel** | **63.56% → 59.01%** | **72.87% → 70.05%** | 17.78% → 18.42% | **9.88% → 10.22%** |
| PCEN | 52.69% → 35.07% | 66.15% → 53.48% | **19.52% → 21.40%** | 8.04% → 9.20% |

正式零样本推荐保持`Xeno-canto → LogMel → DS-CNN`。LogMel量化后在Xeno和DB3V的绝对Macro-F1仍最高；PCEN虽有较高BirdSet Top-1，但PTQ退化明显。

来源：

- `src/experiments/Feature_comparison_8class/comparison_summary.csv`
- `src/experiments/Feature_comparison_8class/birdset_common_20shot_heldout_summary.csv`
- `src/experiments/ZeroShot_strict_INT8_quantization_8class/summary.csv`

## DB3V严格5/10/20-shot实验

四种策略定义固定如下：

| 策略 | 可训练范围 | BatchNorm | Xeno replay |
|---|---|---|---|
| `head_only` | 最终八类softmax层 | 冻结 | 无 |
| `bn_head` | 全部BatchNorm与最终层 | 训练 | 无 |
| `bn_head_replay` | BN+Head | 训练 | support与类别均衡Xeno切片1:1 |
| `full` | 全部层 | 训练 | 无 |

共完成108个FP32训练。每个“特征×shot”只按三个seed的平均适配分数预先选择策略；该分数只读取support内部验证和Xeno保留率，DB3V test与BirdSet test不参与选型。下表是9组预先选中策略的三seed均值±样本标准差。

| 特征 | shot/策略 | Xeno Macro-F1（FP32→INT8） | DB3V源录音Macro-F1（FP32→INT8） |
|---|---|---:|---:|
| MFCC | 5 / `head_only` | 47.73±0.31% → 47.78±0.31% | 55.17±0.20% → 54.63±0.18% |
| MFCC | 10 / `bn_head_replay` | 49.16±0.49% → 47.01±0.50% | 57.49±0.27% → 56.17±0.63% |
| MFCC | 20 / `bn_head_replay` | 48.39±0.45% → 48.02±0.55% | 57.40±0.86% → 56.99±0.92% |
| LogMel | 5 / `head_only` | 59.59±0.23% → 53.32±0.32% | 67.23±0.48% → 63.79±0.19% |
| LogMel | 10 / `bn_head_replay` | 61.00±0.26% → 59.38±0.50% | 68.12±0.38% → 66.67±0.96% |
| **LogMel** | **20 / `bn_head_replay`** | **61.36±0.21% → 60.01±1.32%** | **68.84±0.68% → 67.73±0.59%** |
| PCEN | 5 / `full` | 62.65±0.82% → 37.15±0.50% | **71.93±0.48%** → 55.20±1.10% |
| PCEN | 10 / `full` | 62.83±0.67% → 32.34±2.42% | 71.78±1.00% → 46.09±6.09% |
| PCEN | 20 / `full` | 61.64±0.65% → 28.08±4.86% | 71.27±0.20% → 37.73±9.40% |

结论：

- 预先选中策略中，FP32目标域最高的是PCEN 5-shot `full`，DB3V为71.93%±0.48%；
- PCEN PTQ严重失稳，不能沿用FP32推荐；
- 当前DB3V严格INT8推荐为LogMel 20-shot `bn_head_replay`，DB3V为67.73%±0.59%，Xeno为60.01%±1.32%；
- 完整36组结果在`aggregate.csv`中。未被预设适配分数选中的策略不能因held-out更高而事后改列推荐。

来源：

- `src/experiments/DB3V_fewshot_ablation_multiseed_8class/{runs.csv,aggregate.csv,experiment_protocol.json}`
- `src/experiments/DB3V_fewshot_ablation_multiseed_8class/selected_cross_domain_aggregate.csv`
- `src/experiments/DB3V_strict_INT8_quantization_8class/{summary.csv,aggregate.csv}`

## BirdSet严格5/10/20-shot实验

BirdSet同样完成108个FP32模型和27个预先选中模型的严格INT8复测。下表为三seed均值±样本标准差。

| 特征 | shot/策略 | Xeno Macro-F1（FP32→INT8） | BirdSet Top-1（FP32→INT8） | 纯单物种Macro-F1（FP32→INT8） |
|---|---|---:|---:|---:|
| MFCC | 5 / `bn_head_replay` | 48.41±0.44% → 48.39±1.59% | 20.06±0.57% → 20.15±0.82% | 13.32±0.56% → 14.45±1.02% |
| MFCC | 10 / `bn_head_replay` | 47.89±0.61% → 47.81±0.71% | 19.88±0.29% → 20.69±0.57% | 13.44±0.25% → 15.20±0.43% |
| **MFCC** | **20 / `bn_head_replay`** | 48.53±0.40% → 47.72±0.95% | 23.89±0.44% → **24.12±1.43%** | 17.57±0.49% → **18.40±1.35%** |
| LogMel | 5 / `bn_head_replay` | 56.10±0.63% → 53.14±1.57% | 20.31±2.17% → 20.22±5.03% | 13.32±2.35% → 13.20±1.49% |
| LogMel | 10 / `bn_head_replay` | 56.11±1.31% → 54.40±1.88% | 20.39±2.02% → 23.09±4.19% | 13.66±2.57% → 14.19±0.66% |
| LogMel | 20 / `head_only` | 57.81±1.24% → **55.62±2.18%** | 18.66±0.91% → 21.18±1.76% | 10.62±0.75% → 12.21±1.24% |
| PCEN | 5 / `bn_head` | 55.27±1.34% → 17.97±2.16% | 36.11±1.11% → 19.78±0.09% | 27.11±0.40% → 4.25±0.08% |
| PCEN | 10 / `bn_head` | 54.76±1.29% → 19.03±1.08% | **37.56±2.08%** → 19.84±0.04% | **28.06±1.60%** → 4.34±0.11% |
| PCEN | 20 / `bn_head` | 57.85±0.06% → 16.85±1.33% | 29.68±0.41% → 19.28±0.18% | 23.42±0.18% → 4.18±0.05% |

结论：

- FP32目标域最优为PCEN 10-shot `bn_head`；
- PCEN的三个BirdSet链路PTQ后均不可部署；
- 只强调BirdSet INT8指标时，MFCC 20-shot `bn_head_replay`最佳；
- 同时强调Xeno和DB3V时，LogMel 20-shot `head_only`更均衡，其量化后完整DB3V源录音级Macro-F1为67.34%。

来源：

- `src/experiments/BirdSet_fewshot_ablation_multiseed_8class/{runs.csv,aggregate.csv,experiment_protocol.json}`
- `src/experiments/BirdSet_strict_INT8_quantization_8class/{summary.csv,aggregate.csv}`
- `src/experiments/BirdSet_common_test_8class/test_protocol.json`

## 严格INT8接口

| 特征 | 特征数组 | 模型输入 |
|---|---|---|
| MFCC | `float32 [N,32,13]` | `int8 [N,32,13,1]` |
| LogMel | `float32 [N,32,40]` | `int8 [N,32,40,1]` |
| PCEN | `float32 [N,32,40]` | `int8 [N,32,40,1]` |

每个模型从TFLite读取自己的input/output scale与zero-point。57个模型全部为整数图：int8输入/输出、0浮点张量、无`QUANTIZE`/`DEQUANTIZE`；int32偏置和累加器属于正常整数卷积实现。单模型约49,968–49,984 bytes。

量化校准只读取Xeno训练特征及匹配链路的support；Xeno验证和两个外部test均不参与。量化结果不反向选择FP32特征、shot、策略或epoch。

## 当前推荐链路

| 场景 | 推荐 | 关键结果 |
|---|---|---|
| 通用FP32零样本 | Xeno→LogMel→DS-CNN | Xeno 63.56%；DB3V源录音72.87% |
| 通用INT8零样本 | Xeno→LogMel→DS-CNN→INT8 | Xeno 59.01%；DB3V源录音70.05% |
| DB3V严格FP32 | PCEN 5-shot `full` | DB3V 71.93%±0.48%；仅限FP32 |
| DB3V严格INT8 | LogMel 20-shot `bn_head_replay` | DB3V 67.73%±0.59%；Xeno 60.01%±1.32% |
| BirdSet严格FP32 | PCEN 10-shot `bn_head` | Top-1 37.56%±2.08%；纯单物种F1 28.06%±1.60% |
| BirdSet目标域INT8 | MFCC 20-shot `bn_head_replay` | Top-1 24.12%±1.43%；纯单物种F1 18.40%±1.35% |
| BirdSet跨域均衡INT8 | LogMel 20-shot `head_only` | Xeno 55.62%±2.18%；DB3V源录音67.34% |

## 嵌入式部署证据

历史完整板端运行保留在固定commit上，用于证明工具链曾完整工作，但不能证明本轮刷新模型包已经复刻。

| 项目 | 历史完整运行 | 当前状态 |
|---|---|---|
| 固件/模型 | LED_TEST `073a1f1`；训练仓库`42ead2e` | LED_TEST `134fe68`；本轮模型工件尚未完成板端刷新 |
| 模型烧录 | 57/57通过 | 新MFCC固件构建与烧录验证通过 |
| PCM预测 | 10,944次 | 当前包待UART恢复后复测 |
| TFLite–TFLM输出 | 570组，0不一致，最大0 LSB | 待刷新 |
| 特征一致性 | 历史运行没有形成当前严格特征张量证据 | **受阻：VCP TX未到达LPUART RX** |
| 功耗 | 未测 | **受阻：无电流测量设备** |

历史中位数：

| 特征 | 前端延迟 | 推理延迟 | tensor arena使用 |
|---|---:|---:|---:|
| MFCC | 81.716 ms | 422.868 ms | 27,332 B |
| LogMel | 67.291 ms | 2,112.625 ms | 79,172 B |
| PCEN | 252.825 ms | 2,106.398 ms | 79,172 B |

静态资源测量：

| 固件 | Flash text+data | 静态RAM data+bss |
|---|---:|---:|
| zero-shot MFCC | 179,856 B | 176,816 B |
| zero-shot LogMel | 172,456 B | 176,800 B |
| zero-shot PCEN | 176,272 B | 176,800 B |

链接脚本容量为Flash 524,288 B、RAM1 196,608 B、RAM2 65,536 B；当前链接结果未使用RAM2。静态RAM包含配置的98,304 B tensor arena，不能与arena使用峰值再次相加。

板端输出科学指标必须使用与桌面完全相同的张量test；原始PCM probe只有来源标签，未经逐段人工确认，不作为独立准确率证据。详细状态、commit和阻塞原因见
`src/experiments/Board_replay_8class/deployment_evidence.json`。

功耗采集协议：

1. 外部功耗仪导出`timestamp_s,current_ma[,voltage_v]`；
2. 用`chain_id,phase,start_s,end_s,iterations`标记空闲、前端和推理区间；
3. 运行`analyze_board_power.py`计算均值、标准差、每次能量与输入SHA-256；
4. 报告电压、采样率、重复次数和是否扣除空闲基线。

## 板端复刻包

项目根目录`board_replay_testset/`由脚本生成并被Git忽略，包含：

- `tensors/full`：桌面57模型评估使用的完整数组；
- `tensors/probe`：保持录音/片段分组的快速迁移子集；
- `models`：57个严格INT8模型与输入接口；
- `raw_pcm_probe`：用于板端前端与端到端通路检查的PCM16 probe；
- manifest、标签映射与SHA-256。

`hardlink`模式避免在本机重复占用约3 GB；复制到Linux服务器后应重新核对清单哈希。

## 复现命令

```powershell
# DB3V源录音级划分审计
& .\.venv\Scripts\python.exe src\experiments\audit_db3v_source_split.py

# 重新汇总108个DB3V运行，并对选中策略强制做BirdSet复测
& .\.venv\Scripts\python.exe src\experiments\run_db3v_ablation_multiseed.py --force-cross-domain

# 三个量化家族
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families zero_shot `
  --output-dir src\experiments\ZeroShot_strict_INT8_quantization_8class

& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families db3v_strict_fewshot `
  --output-dir src\experiments\DB3V_strict_INT8_quantization_8class

& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families birdset_strict_fewshot `
  --output-dir src\experiments\BirdSet_strict_INT8_quantization_8class

# 研究证据矩阵与板端复刻包
& .\.venv\Scripts\python.exe src\experiments\build_research_evidence_matrix.py
& .\.venv\Scripts\python.exe src\dataset_processing\prepare_board_replay_testset.py --materialize hardlink
```

Linux板端运行命令与串口协议见
`src/experiments/Board_replay_8class/README.md`和生成包内`README.md`。

## 结果来源索引

- 零样本FP32：`src/experiments/Feature_comparison_8class/`
- DB3V严格FP32：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/`
- BirdSet严格FP32：`src/experiments/BirdSet_fewshot_ablation_multiseed_8class/`
- 零样本严格INT8：`src/experiments/ZeroShot_strict_INT8_quantization_8class/`
- DB3V严格INT8：`src/experiments/DB3V_strict_INT8_quantization_8class/`
- BirdSet严格INT8：`src/experiments/BirdSet_strict_INT8_quantization_8class/`
- DB3V划分审计：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/split_audit.json`
- 总证据矩阵：`src/experiments/research_evidence_matrix.json`
- 板端证据：`src/experiments/Board_replay_8class/deployment_evidence.json`
- 数据集官方来源与许可：见上文“数据来源、许可与实验角色”。

## 指标解释

- **Accuracy**：整体正确比例，受类别不平衡影响。
- **Balanced Accuracy**：各类别召回率的算术平均。
- **Macro-F1**：每类F1等权平均，是单标签八类任务主指标。
- **Top-3**：真实类别是否位于概率最高三类。
- **BirdSet Top-1任意目标**：最高概率类别是否命中multi-hot真实标签中的任一类；不等同单标签Accuracy。
- **一秒切片、八秒块、原始来源录音**：三个相关但不独立的DB3V层级。当前只以最上层原始来源录音作主统计单位。
- **均值±标准差**：严格小样本表均使用种子42、123、2026的样本标准差（`ddof=1`）；不包含重新抽取外部support的方差。
