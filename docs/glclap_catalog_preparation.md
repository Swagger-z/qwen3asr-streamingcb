# GLCLAP 词频表、训练负词与 10k 评测库构建

本项目不提交语料、实体标注、生成后的 catalog 或 embedding index。下面的
数据文件都需要在实验机上构建，并由构建报告记录来源、过滤条件和随机种子。

## 三类数据不能混为一谈

| 数据 | 作用 | 是否需要实体标注 |
|---|---|---|
| `word_freq.txt` | 提供普通词、领域词和其他实体作为 distractor | 否 |
| `zh_train_10k.jsonl` | GLCLAP 每个 batch 共享采样的训练负词池 | 否 |
| `aishell_ner_10k.jsonl` | 检索索引使用的目标实体 + distractor 总表 | 目标部分需要 |

HKUST、MagicData 等其他语料的词频表可以混合，但只能把它们视为 distractor
来源。AISHELL-NER 的目标实体、别名和 `target_hotword_ids` 仍必须来自标注。

## 输入格式

词频表是 UTF-8 文本，每行最后一列为正整数频次：

```text
就是 10998
你 41277
参观 16
那边 1084
```

解析器使用最后一个空白字段作为频次，合并重复词和跨语料计数，然后执行：

- NFKC 规范化并移除空白；
- 默认只保留 2--8 个汉字的词；
- 过滤单字、高于长度上限的词、英文和标点；
- 按频率排名分成 10 桶后等量、确定性采样，避免 top-10k 被高频功能词占满；
- 用文本 SHA-256 前 16 位构造稳定 ID，并在 metadata 中保存各来源频次。

AISHELL-NER 官方 transcript 已直接标注实体，例如：

```text
BAC009S0764W0127 <中原地产>首席分析师[张大伟]说
```

stage1b 解析 `<ORG>`、`[PER]`、`(LOC)` marker，确定性派生标准 catalog：

```json
{"catalog_version":"aishell-ner-test-targets-v1","id":"aishell-ner-...","text":"中原地产","aliases":[],"language":"zh","weight":1.0,"metadata":{"source":"Alibaba-NLP/AISHELL-NER","entity_types":["ORG"],"split":"test"}}
```

以及保留 mention 标签的 manifest：

```json
{"utt_id":"BAC...","audio":"/abs/BAC....wav","text":"中原地产首席分析师张大伟说","target_hotword_ids":["aishell-ner-..."],"entities":[{"mention_id":"BAC...#entity-00","hotword_id":"aishell-ner-...","text":"中原地产","entity_type":"ORG","char_start":0,"char_end":4,"occurrence_index":0}]}
```

转换器不从词表匹配或生成实体标签。评测 catalog 构建器验证所有 ID，并从
distractor 中排除 gold surface forms 及完整 test transcript 的 2--8 字子串。

## 隔离构建训练池与评测库

训练负词池只读取 HKUST、MagicData 等词频文件，不允许读取 AISHELL-NER 的
目标实体或评测转写：

```bash
python scripts/build_glclap_training_pool.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --output data/hotwords/zh_train_10k.jsonl \
  --report data/hotwords/training_pool_report.json \
  --size 10000 \
  --seed 42
```

评测库是独立的数据准备步骤；它读取 AISHELL-NER gold 标注，只供建评测索引
和计算指标使用：

```bash
python scripts/build_glclap_evaluation_catalog.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --target-catalog data/aishell_ner/targets_test.jsonl \
  --eval-manifest data/aishell_ner/test.jsonl \
  --output data/hotwords/aishell_ner_10k.jsonl \
  --report data/hotwords/evaluation_catalog_report.json \
  --size 10000 \
  --seed 42
```

两个 report 分别记录来源统计、过滤项、最终大小、随机种子和是否使用评测
标注。旧的 `build_glclap_catalogs.py` 仅为兼容已有调用保留，正式 `run.sh`
不再使用。若过滤后不足所需数量，命令会失败，不会用重复项静默补足。

AISHELL-NER 原始文件到这两个 gold 文件的转换见
[`aishell_ner_preparation.md`](aishell_ner_preparation.md)；所有 stage 的输入、
输出、生产者和依赖关系见
[`glclap_stage_data_contracts.md`](glclap_stage_data_contracts.md)。

## 训练时的假负例保护

训练入口不仅排除当前 epoch 采样到的 local positive，还会从本 batch 的完整
transcript 中生成全部 2--8 字连续子串，并把这些真实出现的词全部加入排除集。
因此词频池中的“参观”若出现在当前音频中，不会因为本轮恰好采样了另一个
local positive 而被错误当作负例。

四个 GLCLAP 配置中的 `training.local_min_chars` 和
`training.local_max_chars` 同时控制 local positive 与假负例排除范围，默认
分别为 2 和 8。

## 构建边界压力 manifest

stage1b 的 `test_entities.jsonl` 已包含 gold entity text/type/character span；aligner
只为每个 mention 补充 `hotword_start_sec` 和 `hotword_end_sec`：

```bash
AISHELL_NER_ENTITY_MANIFEST=data/aishell_ner/test_entities.jsonl \
AISHELL_NER_TARGET_CATALOG=data/aishell_ner/targets_test.jsonl \
AISHELL_NER_ALIGNED_MANIFEST=data/aishell_ner/test_aligned.jsonl \
bash run_aligner.sh all
```

默认模型为 `Qwen/Qwen3-ForcedAligner-0.6B`。一条音频有多个实体时按 mention
拆分；同一 surface form 重复出现时使用 gold `occurrence_index` 分别解析。完整
item timestamp 另存为 trace，stage2 只消费聚合后的目标 span。forced aligner
不生成实体标签，也不进入在线检索链路。

对齐校验通过后再生成七种边界位置：

```bash
python scripts/build_boundary_stress.py \
  --manifest data/aishell_ner/test_aligned.jsonl \
  --output-dir data/aishell_ner/boundary_wav \
  --output-manifest data/aishell_ner/test_boundary.jsonl \
  --chunk-ms 2000
```

该命令生成 Center、B-400/B-200/B-100、Cross-25/50/75 七种前置静音变体。
生成的 `test_boundary.jsonl` 可直接交给 `decode_streaming_retrieval`。

## 数据泄漏约束

- stage1a 不得读取 AISHELL-NER test transcript、catalog 或实体标签，也不得根据
  test gold 列表反向过滤训练负词池。
- 训练负词池与 test entity surface form 的词面重合只能在训练完成后审计并报告，
  不能据此改写训练输入；文本在推理时本就作为 catalog key 提供。
- 评测 catalog 中普通词可以作为 distractor，但出现在完整 test transcript 中的
  词必须从 distractor 中排除，避免把真实语音内容误计为 false alarm。
- 保存所有输入 SHA-256、构建报告、catalog 和配置快照；数据产物不提交 GitHub。
