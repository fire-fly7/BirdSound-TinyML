# BirdSet统一公共测试集

当前零样本、DB3V适配、BirdSet适配、FP32和严格INT8的所有正式BirdSet横向结果，
统一使用 `birdset_ssw_common_20shot_heldout_v1`。

固定规格：

- 来源：BirdSet SSW `test_5s` shards 1–4；
- 划分：种子42，以原始长录音分组，排除嵌套20-shot support；
- 测试样本：197条长录音、18,265个五秒片段、91,325个不重叠一秒窗口；
- 采样率：16 kHz；
- 每个五秒片段固定产生5个一秒窗口；
- 三种特征使用完全相同的片段身份；
- 三种特征的5/10/20-shot support录音身份一致，规模分别为
  10条/684片段、11条/699片段、14条/1,413片段，且严格满足
  `5-shot ⊂ 10-shot ⊂ 20-shot`；
- support/test原始长录音重叠为0；
- 样本身份SHA-256：
  `1f3a5982558abd0e7dbfe34556410719971b24ac60d29be0166226c892ec056e`。

选择该集合是因为BirdSet的5-shot和10-shot support均嵌套于20-shot support。
排除最大support后，所有shot规模和零样本/DB3V适配模型都能在同一无泄漏测试集上
直接比较。完整211条、5-shot held-out 201条和10-shot held-out 200条报告只保留为
历史诊断，不再进入当前排名。

已知限制：`Setophaga_ruticilla` 在完整筛选集仅有同一support长录音中的两个正片段，
因此公共test中该类无正片段；纯单物种Macro-F1按有支持的7类计算。

文件：

- `test_protocol.json`：样本规格、三种特征目录、test/support身份一致性、
  support嵌套关系、身份哈希及泄漏核验；
- `../birdset_test_protocol.py`：协议生成和强制校验实现；
- `../Feature_comparison_8class/birdset_common_20shot_heldout_summary.csv`：
  三条零样本FP32结果。

从仓库根目录重新核验：

```powershell
& .\.venv\Scripts\python.exe src\experiments\birdset_test_protocol.py
```
