# 板端复刻训练端测试

本实验用于检测同一个严格 INT8 TFLite 从桌面 LiteRT 移植到 STM32
TFLite Micro 后是否保持等价。它复用训练端已经使用的测试样本、标签和聚合粒度，
不把这些样本重新解释为新的独立泛化测试集。

## 当前证据状态

2026-07-29 的历史实板运行已经完成旧模型包 57/57 个固件烧录、10,944 条
PCM16 推理和 570 次桌面 `BUILTIN_REF`/TFLM 输出对拍；预测不一致为 0，最大
输出误差为 0 LSB。该证据固定绑定板端提交
`073a1f1a40d689448b332bac717961475315bebf` 和训练端提交
`42ead2e614e3f40afc7da29e660fc94609d10817`，只能证明这套旧工件的移植链路，
不能替代当前按 DB3V 原始源录音重新划分后的模型包复测。

| 嵌入式证据 | 当前状态 | 可用结论/下一步 |
|---|---|---|
| 输出一致性 | 历史完整，当前模型待复测 | 历史 570 次对拍为 0 mismatch、0 LSB；修复 UART 后对新包重跑 |
| 延迟 | 历史完整，当前模型待复测 | 历史 MFCC/LogMel/PCEN 的前端中位数为 81.716/67.291/252.825 ms，推理中位数为 422.868/2112.625/2106.398 ms |
| Flash | 已测 | 三个代表固件分别使用 179,856/172,456/176,272 byte（`text+data`） |
| 静态 RAM | 已测 | 分别使用 176,816/176,800/176,800 byte（`data+bss`）；RAM2 未使用 |
| Arena 高水位 | 历史完整 | 27,284–79,172 / 98,304 byte |
| 板端特征一致性 | 阻塞 | 新 MFCC 固件已构建、烧录并校验，但 LPUART RX 收不到 `/dev/ttyACM0` 数据；先检查 VCP TX/跳线/线缆路由，再抓取特征 |
| 功耗 | 阻塞 | 当前没有电流探头；ST-Link 目标电压不是功耗。接入功率计后用 `analyze_board_power.py` 统计 |

机器可读的证据、提交指纹、资源占用和阻塞原因保存在
[`deployment_evidence.json`](deployment_evidence.json)。阻塞项不是“实验通过”，README
中的部署结论必须持续保留这一限定。

## 测试范围

| 范围 | 完整样本 | 原始聚合粒度 | 适用链路 |
|---|---:|---|---|
| Xeno-canto validation | 3,652 个一秒片段，480 条录音 | 片段级、录音级 | 全部 57 个模型 |
| BirdSet common 20-shot held-out | 91,325 个一秒片段，18,265 个 clip，197 条长录音 | 多标签片段级、5 秒 clip 级、全局单标签 clip 级 | 全部 57 个模型 |
| DB3V full | 85,264 个一秒片段，10,658 条八秒录音 | 区域片段/录音级、合并片段/录音级 | 零样本和 BirdSet few-shot 模型 |
| DB3V common 20-shot held-out | 61,280 个一秒片段，7,660 条八秒块、970 条原始源录音 | 区域片段/八秒块/原始源录音级、合并对应三级 | DB3V few-shot 模型 |

三种特征的样本身份、标签和 recording/clip 索引已经逐项核对一致。BirdSet
固定 held-out 中没有 `Setophaga_ruticilla` 正样本；DB3V 固定 held-out 的区域 2
没有标签 2 和 4。这是原测试范围本身的稀疏性，不能用其他区域样本补齐后仍称为
同一测试。

## 两阶段移植判据

第一阶段直接发送训练端特征张量：

`训练端 float32 特征 -> 板端输入量化 -> TFLM INT8 推理`

LED_TEST 的 `sweep --tflite` 会保存板端 `raw_output_int8` 和桌面
`reference_raw_int8`。逐样本预测必须一致，最大输出误差必须不超过 1 LSB。
这一步隔离模型、量化、串口编码和 TFLM 内核。

第二阶段发送相同来源的一秒 PCM16：

`PCM16 -> 板端 MFCC/LogMel/PCEN -> 输入量化 -> TFLM`

生成的 raw probe 有 192 条 WAV，八类各 24 条。全部是单声道、16 kHz、
PCM16、16,000 帧，清单同时保存 WAV 和 PCM payload SHA-256。它们保留原
Xeno-canto/DB3V 来源标签，未做人耳单物种复核，所以必须使用
`source-label` 策略，`scientific_metrics_valid=false`。

相同 PCM 在训练端重建特征后，相对原测试数组的最大绝对误差为：

| 特征 | 最大绝对误差 |
|---|---:|
| MFCC | 0.00006103515625 |
| LogMel | 0.0000152587890625 |
| PCEN | 0.0000007152557373 |

以上三项是桌面端从 WAV 重建特征与保存数组的误差，并不是板端特征对拍结果。
严格板端特征一致性必须使用当前固件的 `audio-run --dump-feature` 实际导出张量，
再与同一窗口的训练数组比较；不得从最终分类一致性反推特征已经一致。

UART 路由修复后的最小复测命令为：

```bash
python3 tools/serial_model_client.py --port /dev/ttyACM0 info > info.json
python3 tools/serial_model_client.py --port /dev/ttyACM0 audio-run \
  --input /path/to/aligned.wav --dump-feature board_feature.npy
python3 tools/compare_frontend_features.py \
  --board board_feature.npy \
  --reference /path/to/aligned_test_data.npy \
  --reference-index 0 \
  --info info.json \
  --max-abs-limit 0.001 \
  --output feature_parity.json
```

其中 `0.001` 是复测前固定的 float 上限，严格判据还要求模型输入量化尺度上
0 个不一致元素、最大误差 0 LSB；必须同时保存模型上报的 input
scale/zero-point，不能只写一个未绑定量化参数的容差。

## Flash、RAM 与功耗口径

STM32L552 链接脚本声明 Flash 512 KiB、RAM1 192 KiB、RAM2 64 KiB。保留 ELF
用同一 `arm-none-eabi-size` 工具链得到：

| 代表固件 | Flash `text+data` | Flash占比 | 静态RAM `data+bss` | RAM1占比 |
|---|---:|---:|---:|---:|
| MFCC | 179,856 B | 34.30% | 176,816 B | 89.93% |
| LogMel | 172,456 B | 32.89% | 176,800 B | 89.93% |
| PCEN | 176,272 B | 33.62% | 176,800 B | 89.93% |

静态 RAM 已包含 98,304-byte Tensor Arena 的预留，Arena 实际高水位另由板端
allocator 上报；两者不能相加。RAM2 当前未放置段，因此不能把总 256 KiB 当成
现有固件可连续使用的 RAM1 余量。

功耗必须从开发板供电路径串入电流测量设备，并至少分别采集 `idle`、`frontend`、
`inference` 和 `end_to_end`，每个代表链路预热后重复不少于 30 次。功率计导出
`timestamp_s,current_ma[,voltage_v]`，同步区间按
[`power_intervals_template.csv`](power_intervals_template.csv) 记录，然后执行：

```bash
python3 src/experiments/analyze_board_power.py \
  --samples power_samples.csv \
  --intervals power_intervals.csv \
  --output power_report.json \
  --output-csv power_intervals_result.csv
```

报告同时给出总能量和扣除 idle 基线后的动态能量；如果没有 `idle` 区间或显式
baseline，动态能量会保持为空，不会伪造。

## 参考运行时

原严格 INT8 报告使用 `batch=128` 和默认 TFLite 解释器；LED_TEST 的可提交
对拍固定使用 TensorFlow 2.19、`batch=1` 和 `BUILTIN_REF`。量化内核路径不同
会导致少量 INT8 输出变化。因此：

- 移植是否等价，以同一 CSV 内的 `BUILTIN_REF` 和 TFLM 逐 LSB 对拍为准；
- 旧 batch-128 指标仍保留并报告差值，但不作为板端逐 LSB 真值；
- 完整测试必须继续使用原 recording/clip 聚合，不能只比较单片段 accuracy。

## 生成与运行

在训练仓库生成可复制的数据包：

```powershell
& .\.venv\Scripts\python.exe `
  src\dataset_processing\prepare_board_replay_testset.py `
  --materialize hardlink
```

生成目录 `board_replay_testset/` 已由 Git 忽略。`hardlink` 在本机不重复占用
约 3 GB；把目录复制到 Linux 时会自然成为普通文件。

Linux 上先跑核心模型 probe：

```bash
python3 src/experiments/run_board_replay.py \
  --led-test /srv/LED_TEST \
  --pack /srv/led-firmware-pack \
  --package board_replay_testset \
  --port /dev/ttyACM0 \
  --tier probe \
  --scope core
```

核心模型通过后改为 `--scope all`。最终复算完整指标时使用
`--tier full --mode native`。运行器会逐模型核对 `INDEX.csv`、TFLite
SHA-256、板端 INFO 标签顺序和特征类型，然后烧录、执行 parity、回放
Xeno-canto/BirdSet/DB3V，并调用
[`evaluate_board_replay.py`](../evaluate_board_replay.py) 生成对应粒度的指标。

## 实现与板端来源

- 数据包生成：[`prepare_board_replay_testset.py`](../../dataset_processing/prepare_board_replay_testset.py)
- Linux 自动运行：[`run_board_replay.py`](../run_board_replay.py)
- 指标恢复：[`evaluate_board_replay.py`](../evaluate_board_replay.py)
- 板端测试规格：
  [LED_TEST BOARD_BENCHMARK_TESTSET.md](https://github.com/fire-fly7/BirdSound-STM32/blob/073a1f1a40d689448b332bac717961475315bebf/docs/BOARD_BENCHMARK_TESTSET.md)
- 板端串口协议：
  [LED_TEST serial_model_client.py](https://github.com/fire-fly7/BirdSound-STM32/blob/073a1f1a40d689448b332bac717961475315bebf/tools/serial_model_client.py)
- 板端全模型执行器：
  [LED_TEST run_board_benchmark.py](https://github.com/fire-fly7/BirdSound-STM32/blob/073a1f1a40d689448b332bac717961475315bebf/tools/run_board_benchmark.py)
