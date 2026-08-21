# GLCLAP 实验逐 Stage 数据契约

本文档是 `run.sh` 的强制数据接口说明。每个文件均标明来源、schema、消费者和
是否依赖前序 stage。路径可以通过同名环境变量覆盖；数据、模型、索引和输出不提交
Git。

## 先区分八类数据

| 类别 | 例子 | 训练使用 | 评测使用 | 生成方式 |
|---|---|---:|---:|---|
| 原始 ASR 训练清单 | `data/aishell1/train.jsonl` | 是 | 否 | 从 AISHELL-1 train 转换 |
| 训练负词候选源 | HKUST/MagicData `word_freq.txt` | 是 | 可作 distractor 源 | 外部语料统计 |
| 训练负词池 | `zh_train_10k.jsonl` | 是 | 否 | stage1a |
| AISHELL-NER gold transcript | `aishell_ner_transcript.test.txt` | 否 | 是 | 数据集自带 PER/LOC/ORG marker |
| 评测目标注册表 | `targets_test.jsonl` | 否 | 是 | stage1b 解析 gold marker |
| 全量评测清单 | `test.jsonl` | 否 | 是 | stage1b，保留无实体 utterance |
| 实体边界清单 | `test_entities.jsonl` | 否 | 是 | stage1b，仅实体 utterance/mention |
| 评测检索库 | `aishell_ner_10k.jsonl` | 否 | 是 | stage1c：gold target + distractor |

`target-catalog` 是 AISHELL-NER gold surface form 的内部注册表，不是训练时抽取的
local positive。默认训练 positive 仍由 `AISHELL1_TRAIN_MANIFEST.target` 按 epoch
确定性随机截取。评测标注不得输入 stage1a。

## 原始 ASR manifest 规范

数据集提供的所有原始 ASR JSONL 均采用以下字段：

```json
{"key":"BAC009S0002W0122","source":"/data/aishell/wav/BAC009S0002W0122.wav","target":"而对楼市成交抑制作用最大的限购"}
```

程序为兼容已有内部派生文件，也接受 `utt_id/audio/text` 别名；新生成的 AISHELL-NER
manifest 同时写出两套字段，但以 `key/source/target` 为规范接口。


## 外部输入 schema
### 1. `HKUST_WORD_FREQ` / `MAGICDATA_WORD_FREQ`

来源：外部、独立准备。编码 UTF-8，每行最后一列是正整数频次：

```text
参观 16
语音识别 892
人工智能 615
```

约束：默认只保留 2--8 个汉字；重复项合并；单字、英文、标点和非正频次被过滤
或报错。它们仅是训练负例和评测 distractor 的候选源，不是实体真值。

### 2. `AISHELL1_TRAIN_MANIFEST`

来源：外部、独立准备。每行一个 AISHELL-1 train utterance：

```json
{"utt_id":"BAC009S0002W0122","audio":"/data/aishell1/wav/u.wav","text":"今天前往人工智能产业园参观"}
```

必需字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `utt_id` | string | 全局唯一 utterance ID |
| `audio` | string | 可读取的音频路径，训练入口重采样/校验到 16 kHz |
| `text` | string | 完整人工转写，global positive 来源 |

默认 hybrid 协议每个 epoch 从 `text` 确定性抽一个 2--8 字连续子串作为 local
positive。该文件不引用 AISHELL-NER test 标注。

### 3. `AISHELL_NER_ANNOTATED_TRANSCRIPT` / `AISHELL_NER_WAV_ROOT`

来源：外部、独立准备。前者是 AISHELL-NER 官方 split 文件，后者是对应的
AISHELL-1 WAV 根目录：

```text
BAC009S0764W0127 <中原地产>首席分析师[张大伟]说
BAC009S0764W0130 (北京)仅新增住宅土地供应十宗
```

`[实体]`、`(实体)`、`<实体>` 分别是数据集自带的 PER、LOC、ORG gold 标记。
stage1b 只解析标记、去除标记并绑定 WAV，不做词表匹配或 NER 推断。

### 4. Stage1b 派生的三个 gold 视图

`AISHELL_NER_TARGET_CATALOG` 每个唯一 surface form 一条：

```json
{"catalog_version":"aishell-ner-test-targets-v1","id":"aishell-ner-...","text":"中原地产","aliases":[],"language":"zh","weight":1.0,"metadata":{"source":"Alibaba-NLP/AISHELL-NER","split":"test","entity_types":["ORG"],"mention_count":3}}
```

`AISHELL_NER_EVAL_MANIFEST` 保留全部 test utterance，包括无实体记录，供完整评测：

```json
{"utt_id":"BAC...","audio":"/abs/BAC....wav","text":"中原地产首席分析师张大伟说","target_hotword_ids":["aishell-ner-..."],"entities":[{"mention_id":"BAC...#entity-00","hotword_id":"aishell-ner-...","text":"中原地产","entity_type":"ORG","char_start":0,"char_end":4,"occurrence_index":0}]}
```

`AISHELL_NER_ENTITY_MANIFEST` 使用相同 schema，但过滤掉无实体 utterance，供
forced aligner 和边界压力实验使用。核心字段：

| 字段 | 位置 | 含义 |
|---|---|---|
| `target_hotword_ids` | utterance | 本句涉及的唯一 catalog ID |
| `entities[].mention_id` | mention | 不去重的稳定 mention ID |
| `entities[].hotword_id` | mention | 指向 target catalog |
| `entities[].entity_type` | mention | `PER`/`LOC`/`ORG` |
| `entities[].char_start/end` | mention | marker-free transcript 的半开字符区间 |
| `entities[].occurrence_index` | mention | 同 surface form 在本句中的第几次出现，从 0 开始 |

三个文件都由现有 gold 标注确定性派生，不是 Qwen aligner 的输出，也不用于默认
训练 stage3--5。

## 主流水线依赖图

```text
word_freq.txt ────────────────┬─ stage1a ─> zh_train_10k.jsonl ─┬─ stage3
                              │                                  ├─ stage4
                              │                                  └─ stage5
AISHELL-NER tagged transcript ── stage1b ─┬─> targets_test.jsonl ─┐
AISHELL-1 test WAV ───────────────────────┼─> test.jsonl ─────────┴─ stage1c ─> aishell_ner_10k.jsonl ─> stage6
                                         └─> test_entities.jsonl ─> run_aligner.sh
                                                                    └─> test_aligned.jsonl ─> stage2
                                                                                             └─> test_boundary.jsonl
checkpoints + evaluation catalog ─> stage6 index
checkpoint + index + boundary manifest ─> stage7 retrieval JSONL ─> stage8 metrics
```

## Stage 0：环境与复现快照

输入：无数据输入。需要 Python 3.10+、固定 Qwen/vLLM/Transformers、Linux CUDA。

输出：

- `outputs/glclap/run_metadata/git_commit.txt`
- `outputs/glclap/run_metadata/environment.freeze.txt`
- `outputs/glclap/run_metadata/nvidia-smi.txt`（若存在）

依赖：独立，可先运行。不会生成任何 manifest。

## Stage 1：训练池、gold 转换与评测库（三个隔离子步骤）

### Stage 1a：训练负词池

输入：仅两个 `word_freq.txt`。不读取 target catalog 或 eval manifest。

命令：

```bash
python scripts/build_glclap_training_pool.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --output data/hotwords/zh_train_10k.jsonl \
  --report data/hotwords/training_pool_report.json \
  --size 10000 --seed 42
```

输出 JSONL：

```json
{"catalog_version":"zh-train-neg-v2","id":"neg-...","text":"语音识别","aliases":[],"language":"zh","weight":1.0,"metadata":{"role":"train_negative","total_frequency":892,"source_frequencies":{"hkust":400,"magicdata":492}}}
```

消费者：stage3--5。该池在每个 batch 中采 4095 个共享 negatives。训练代码还会
排除当前 batch transcript 中全部 2--8 字真实子串，避免假负例。

### Stage 1b：解析 AISHELL-NER gold marker

输入：官方 tagged transcript + AISHELL-1 test WAV，不依赖 stage1a。

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

输出是已有标签的三个内部视图和审计报告；`entity_labels_inferred=false`。

### Stage 1c：评测 target+distractor catalog

输入：两个 `word_freq.txt` 和 stage1b 的 target catalog/full eval manifest。与
stage1a 没有数据依赖。

```bash
python scripts/build_glclap_evaluation_catalog.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --target-catalog data/aishell_ner/targets_test.jsonl \
  --eval-manifest data/aishell_ner/test.jsonl \
  --output data/hotwords/aishell_ner_10k.jsonl \
  --report data/hotwords/evaluation_catalog_report.json \
  --size 10000 --seed 42
```

输出包含全部 gold targets 和补足到 10k 的 distractors。distractor 会排除 target
surface forms 及全量 test transcript 中实际出现的 2--8 字子串。消费者：stage6
建库、stage7/8 评测；不用于训练。

## 独立对齐流水线：`run_aligner.sh`

输入：stage1b 的 target catalog + entity-only manifest + 原始音频 +
`Qwen/Qwen3-ForcedAligner-0.6B`。

输出 `AISHELL_NER_ALIGNED_MANIFEST`：

```json
{"utt_id":"BAC...__mention_...","source_utt_id":"BAC...","audio":"/abs/u.wav","text":"中原地产首席分析师张大伟说","target_hotword_ids":["aishell-ner-..."],"aligned_hotword_id":"aishell-ner-...","aligned_mention_id":"BAC...#entity-00","hotword_start_sec":0.731,"hotword_end_sec":1.284}
```

每个 gold mention 拆成一条记录；重复 surface form 按 `occurrence_index` 对齐。该
步骤依赖 stage1b，但不修改实体标签，也不进入在线运行链路。

## Stage 2：边界压力样本

输入：`AISHELL_NER_ALIGNED_MANIFEST`。前序依赖是 stage1b 后运行的独立 aligner。

输出：每个对齐记录生成 center、b-400、b-200、b-100、cross-25、cross-50、
cross-75 七条 PCM16 WAV 与 `test_boundary.jsonl`。每条输出继承 target ID，并新增：

```json
{"boundary_group":"cross-50","leading_silence_sec":0.412,"word_start_sec":1.143,"word_end_sec":1.696}
```

消费者：stage7。stage2 不训练模型。

## Stage 3：global-only CLAP 基线

输入：

- `AISHELL1_TRAIN_MANIFEST`（外部）
- `NEGATIVE_CATALOG`（stage1a）
- Qwen3-ASR 权重（外部）
- `configs/glclap/global_only_clap.yaml`（仓库）

输出：`outputs/glclap/global_only/{config.snapshot.json,train.jsonl,last.pt}`。
当前 global loss 使用完整 transcript；local weight 为 0。

## Stage 4：冻结 Qwen projector 的 GLCLAP 主系统

输入与 stage3 相同，但配置为 `qwen_post_projector_frozen.yaml`。当前实验协议是：

- global positive：完整 transcript；
- local positive：每 epoch 从 transcript 确定性随机抽 2--8 字子串；
- local negatives：从 stage1a pool 采 4095 个并在 batch 内共享；
- Qwen AuT、原 projector 和 LLM token embedding 冻结，仅训练双 MLP adapter
  与 temperature。

这是项目的 **hybrid Qwen-AISHELL1 协议**，不是 GLCLAP 或 AmphionASR 的严格
复现。输出目录为 `outputs/glclap/frozen/`。

## Stage 5：初始化消融

输入：同 stage4。分别使用 `qwen_projector_warmstart.yaml` 和
`random_projector.yaml`；依赖 stage1a，但不依赖 stage3/4 checkpoint。

输出：`outputs/glclap/{warmstart,random}/last.pt` 与训练日志。

## Stage 6：文本 embedding index

输入：

- stage1b 的 `EVALUATION_CATALOG`
- stage3--5 的各变体 checkpoint
- 对应 config 与 Qwen 权重

输出 NPZ：`keys[variant,512]`（fp16）、`hotword_ids`、`variants`、内嵌 metadata；
另有 `.npz.json`，记录 checkpoint/catalog SHA-256。canonical 与 alias 各编码一次，
同一 ID 在检索时取最大 variant 分数。

## Stage 7：累计音频检索

输入：stage2 boundary manifest + stage6 index + 对应 checkpoint/config。

输出 `boundary_retrieval.jsonl`，每条包含原始真值、`batches`、`final_batch`、
Top-K hits、分阶段延迟、RTF、显存及可选 offline parity。hit schema：

```json
{"hotword_id":"ne-0001","score":0.82,"rank":1,"peak_frame":37,"variant":"商汤科技"}
```

## Stage 8：指标

输入：stage7 各变体 JSONL；global-only 作为配对 baseline。

输出：Recall@1/5/10/20/50、Precision/F1/FAR、MRR、BoundaryPenalty、首次进入
Top-50 chunk、P50/P95 延迟、RTF、显存、offline exact-match 与配对 bootstrap CI。

## Stage 9：测试与产物清单

输入：仓库源码；可选 PyTorch/Qwen/CUDA 环境。输出仅为日志。默认运行全量单测、
100k catalog 和 10k×512 检索压力测试。模型契约测试在没有固定运行时和权重时会
显式 skip。

## 生成顺序

```bash
# 1. 外部准备四类原始输入：word_freq、train manifest、target catalog、eval manifest
# 2. 独立生成参考时间戳
INSTALL_DEPS=1 bash run_aligner.sh all
# 3. 主流水线：stage1a/1b -> stage2 -> train -> index -> decode -> eval
bash run.sh all
```

修改数据 split、catalog size、seed、模型或 chunk size 时必须使用新的
`OUTPUT_ROOT`，并保留所有 report、config snapshot、manifest SHA-256 和 Git commit。
