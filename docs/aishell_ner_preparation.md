# AISHELL-NER gold 标签转换与评测数据契约

本项目默认直接使用 [Alibaba-NLP/AISHELL-NER](https://github.com/Alibaba-NLP/AISHELL-NER)
自带的命名实体标注，不再使用 SeACo-Paraformer 的 `hotword.txt` 构造或反推评测
实体。`prepare_aishell_ner.py` 只做确定性的格式转换和 WAV 路径绑定，报告中的
`entity_labels_inferred` 固定为 `false`。

## 官方输入

AISHELL-NER 提供 train/dev/test 三个带标签 transcript。每行格式为：

```text
UTT_ID TAGGED_TRANSCRIPT
BAC009S0764W0127 <中原地产>首席分析师[张大伟]说
BAC009S0764W0130 (北京)仅新增住宅土地供应十宗
```

三类 gold marker 为：

| 标记 | 实体类型 |
|---|---|
| `[实体]` | `PER`，人名 |
| `(实体)` | `LOC`，地点 |
| `<实体>` | `ORG`，组织机构 |

音频仍来自 AISHELL-1。转换器递归扫描 `--wav-root`，用 WAV 文件名 stem 与
`UTT_ID` 一一匹配；缺失音频、重复 ID、空实体、未闭合或嵌套 marker 均直接失败。

## Stage1b/1c 分离转换

`run.sh stage1` 先处理 dev，再独立处理 test：

dev 用于 checkpoint validation：
```bash
python scripts/prepare_aishell_ner.py \
  --annotated-transcript /data/AISHELL-NER/data/aishell_ner_transcript.dev.txt \
  --wav-root /data/AISHELL-1/wav/dev \
  --target-catalog-output data/aishell_ner/targets_dev.jsonl \
  --eval-manifest-output data/aishell_ner/dev.jsonl \
  --entity-manifest-output data/aishell_ner/dev_entities.jsonl \
  --report data/aishell_ner/dev_preparation_report.json \
  --split dev
```

test 只用于最终评测：
```bash
python scripts/prepare_aishell_ner.py \
  --annotated-transcript /data/AISHELL-NER/data/aishell_ner_transcript.test.txt \
  --wav-root /data/AISHELL-1/wav/test \
  --target-catalog-output data/aishell_ner/targets_test.jsonl \
  --eval-manifest-output data/aishell_ner/test.jsonl \
  --entity-manifest-output data/aishell_ner/test_entities.jsonl \
  --report data/aishell_ner/test_preparation_report.json \
  --split test
```

两步都不调用 ASR、NER 模型或 forced aligner，也不会从外部词表匹配实体。
训练验证只读 `dev_entities.jsonl` 的 `entities[].text`，不会读取 test 产物。

## 三个派生视图

### Gold target catalog

每个唯一实体 surface form 一条记录，供评测检索库和 forced aligner 查表：

```json
{"catalog_version":"aishell-ner-test-targets-v1","id":"aishell-ner-...","text":"中原地产","aliases":[],"language":"zh","weight":1.0,"metadata":{"source":"Alibaba-NLP/AISHELL-NER","split":"test","entity_types":["ORG"],"mention_count":3,"role":"evaluation_target"}}
```

它是 gold 标签的规范化注册表，不是训练时随机抽取的 local positive。

### Full evaluation manifest

`test.jsonl` 保留 test split 的全部 utterance，包括没有实体的句子。它用于构建
评测 catalog、普通词 false-alarm/UWER 对照，以及后续完整 test 评测：

```json
{"utt_id":"BAC...","audio":"/abs/BAC....wav","text":"中原地产首席分析师张大伟说","target_hotword_ids":["aishell-ner-..."],"entities":[{"mention_id":"BAC...#entity-00","hotword_id":"aishell-ner-...","text":"中原地产","entity_type":"ORG","char_start":0,"char_end":4,"occurrence_index":0}]}
```

### Entity-only manifest

`test_entities.jsonl` 只保留至少含一个 gold entity 的 utterance，供
`dev_entities.jsonl` 同样只保留有实体的 dev 语音，训练验证将每条记录中全部不同
`entities[].text` 作为 local positives，用它选择 `best.pt`。

`run_aligner.sh` 和七种 chunk-boundary 压力样本使用。每个 mention 都有独立
`mention_id`；同一句中同一个实体重复两次时不会被去重，forced alignment 会按
`occurrence_index` 分别解析。

## 正确运行顺序

```bash
export AISHELL_NER_ANNOTATED_TRANSCRIPT=/data/AISHELL-NER/data/aishell_ner_transcript.test.txt
export AISHELL_NER_DEV_ANNOTATED_TRANSCRIPT=/data/AISHELL-NER/data/aishell_ner_transcript.dev.txt
export AISHELL_NER_DEV_WAV_ROOT=/data/AISHELL-1/wav/dev
export AISHELL_NER_WAV_ROOT=/data/AISHELL-1/wav/test
export HKUST_WORD_FREQ=/data/hkust/word_freq.txt
export MAGICDATA_WORD_FREQ=/data/magicdata/word_freq.txt

# 生成训练负词池，并把已有 AISHELL-NER gold 标签转换成内部数据视图。
bash run.sh stage0 stage1

# 仅为边界实验补充实体音频时间戳，不生成实体标签。
bash run_aligner.sh all

# 生成边界样本并继续训练、建库、检索和评测。
bash run.sh stage2 stage3 stage4 stage5 stage6 stage7 stage8 stage9
```

AISHELL-NER train split 也带实体标签，可用于后续 Amphion-style
`annotated-hotword positive` 对照；当前默认 GLCLAP-inspired 主配置仍从普通
AISHELL-1 train transcript 随机抽 2--8 字 local positive，二者必须在实验表中
分开报告。
