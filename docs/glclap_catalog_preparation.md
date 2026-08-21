# GLCLAP 词频表、训练负词与 10k 评测库构建

本项目不提交语料、实体标注、生成后的 catalog 或 embedding index。下面的
数据文件都需要在实验机上构建，并由构建报告记录来源、过滤条件和随机种子。

## 三类数据不能混为一谈

| 数据 | 作用 | 是否需要实体标注 |
|---|---|---|
| `word_freq.txt` | 提供普通词、领域词和其他实体作为 distractor | 否 |
| `zh_train_10k.jsonl` | GLCLAP 每个 batch 共享采样的训练负词池 | 否 |
| `aishell1_ne_10k.jsonl` | 检索索引使用的目标实体 + distractor 总表 | 目标部分需要 |

HKUST、MagicData 等其他语料的词频表可以混合，但只能把它们视为 distractor
来源。AISHELL1-NE 的目标实体、别名和 `target_hotword_ids` 仍必须来自标注。

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

目标实体文件继续使用标准 catalog JSONL：

```json
{"catalog_version":"aishell1-ne-target-v1","id":"ne-0001","text":"张江人工智能岛","aliases":["人工智能岛"],"language":"zh","weight":1.0,"metadata":{"entity_type":"poi","split":"test"}}
```

评测 manifest 至少包含：

```json
{"utt_id":"BAC009S0002W0122","audio":"/data/aishell1_ne/test.wav","text":"今天前往张江人工智能岛参观","target_hotword_ids":["ne-0001"]}
```

构建器会验证 manifest 中每个目标 ID 都存在于目标 catalog。评测 distractor
还会排除目标文本、aliases，以及评测 transcript 中实际出现的全部 2--8 字
子串，避免未标注但真实说出的词被错误计为 false alarm。

## 一次构建训练库和评测库

```bash
build_glclap_catalogs \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --negative-output data/hotwords/zh_train_10k.jsonl \
  --target-catalog data/aishell1_ne/targets_test.jsonl \
  --eval-manifest data/aishell1_ne/test.jsonl \
  --evaluation-output data/hotwords/aishell1_ne_10k.jsonl \
  --report data/hotwords/catalog_build_report.json \
  --size 10000 \
  --seed 42
```

也可以在源码目录运行：

```bash
python scripts/build_glclap_catalogs.py ...
```

如果 AISHELL1-NE 目标标注尚未准备好，只传 `--word-freq` 和
`--negative-output` 即可先生成训练负词库。`--evaluation-output` 必须与
`--target-catalog`、`--eval-manifest` 一起使用。

输出的 `catalog_build_report.json` 记录每个来源的总行数、接受/过滤行数、
唯一词数、合并后词数、目标数、最终 catalog 大小和随机种子。若过滤后不足
所需数量，命令会失败，不会用重复项静默补足。

## 训练时的假负例保护

训练入口不仅排除当前 epoch 采样到的 local positive，还会从本 batch 的完整
transcript 中生成全部 2--8 字连续子串，并把这些真实出现的词全部加入排除集。
因此词频池中的“参观”若出现在当前音频中，不会因为本轮恰好采样了另一个
local positive 而被错误当作负例。

四个 GLCLAP 配置中的 `training.local_min_chars` 和
`training.local_max_chars` 同时控制 local positive 与假负例排除范围，默认
分别为 2 和 8。

## 构建边界压力 manifest

先使用离线 forced aligner 为目标热词产生 `hotword_start_sec` 和
`hotword_end_sec`，形成 `test_aligned.jsonl`。forced aligner 不进入在线链路。

```bash
python scripts/build_boundary_stress.py \
  --manifest data/aishell1_ne/test_aligned.jsonl \
  --output-dir data/aishell1_ne/boundary_wav \
  --output-manifest data/aishell1_ne/test_boundary.jsonl \
  --chunk-ms 2000
```

该命令生成 Center、B-400/B-200/B-100、Cross-25/50/75 七种前置静音变体。
生成的 `test_boundary.jsonl` 可直接交给 `decode_streaming_retrieval`。

## 数据泄漏约束

- AISHELL1-NE test 目标文本和 aliases 不得进入训练负词库。
- 如果以后用 HKUST 或 MagicData 做测试，其测试实体必须从本次训练词频池中排除。
- 评测 catalog 中普通词可以作为 distractor，但出现在当前评测 transcript 中的
  词必须标成目标或从 distractor 中排除。
- 保存原始词频文件 SHA-256、构建报告、catalog 和实验配置快照；这些产物不提交
  GitHub，但应随实验归档。
