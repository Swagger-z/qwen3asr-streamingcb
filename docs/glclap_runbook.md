# GLCLAP 完整分阶段实验运行手册

仓库根目录的 `run.sh` 串联了 catalog 构建、边界数据生成、四种模型训练、
索引构建、累计音频检索、评测和压力测试。脚本面向 Linux CUDA 单机，训练
入口仍是单进程；多张 GPU 用于从不同 shell 并行运行不同实验。

每个 stage 的输入文件、逐字段 schema、生产者、消费者和前序依赖统一定义在
[`glclap_stage_data_contracts.md`](glclap_stage_data_contracts.md)。论文采样设置与
当前实现的差异见 [`glclap_reproduction_matrix.md`](glclap_reproduction_matrix.md)。

AISHELL-NER 官方 marker 到内部 gold catalog/manifest 的无推断转换见
[`aishell_ner_preparation.md`](aishell_ner_preparation.md)。

## 外部必备输入

脚本不会下载或提交数据。运行前只需准备以下原始输入：

| 环境变量 | 默认路径 | 必需字段/格式 |
|---|---|---|
| `HKUST_WORD_FREQ` | `data/raw/hkust/word_freq.txt` | 每行 `词语 频次` |
| `MAGICDATA_WORD_FREQ` | `data/raw/magicdata/word_freq.txt` | 每行 `词语 频次` |
| `AISHELL1_TRAIN_MANIFEST` | `data/aishell1/train.jsonl` | `key/source/target` |
| `AISHELL_NER_ANNOTATED_TRANSCRIPT` | `data/raw/AISHELL-NER/data/aishell_ner_transcript.test.txt` | 官方 `UTT_ID TAGGED_TRANSCRIPT` |
| `AISHELL_NER_WAV_ROOT` | `data/raw/AISHELL-1/wav/test` | 可递归扫描的 test WAV 根目录 |

stage1b 直接解析 AISHELL-NER 已有的 `[PER]`、`(LOC)`、`<ORG>` gold marker，
不会根据热词表推断实体。它生成三个内部视图：

| 派生变量 | 默认路径 | 消费者 |
|---|---|---|
| `AISHELL_NER_TARGET_CATALOG` | `data/aishell_ner/targets_test.jsonl` | stage1c、aligner、建库 |
| `AISHELL_NER_EVAL_MANIFEST` | `data/aishell_ner/test.jsonl` | 全量 test 评测，包含无实体句 |
| `AISHELL_NER_ENTITY_MANIFEST` | `data/aishell_ner/test_entities.jsonl` | aligner 与边界压力实验 |

`AISHELL_NER_ALIGNED_MANIFEST` 是 forced aligner 追加声学时间戳后的派生文件。
target catalog 与 manifest 的 ID、mention span 和实体类型均来自同一份官方标注。

对齐是 stage1 之后的独立离线流水线：

```bash
export AISHELL_NER_ENTITY_MANIFEST=/data/aishell_ner/test_entities.jsonl
export AISHELL_NER_TARGET_CATALOG=/data/aishell_ner/targets_test.jsonl
export AISHELL_NER_ALIGNED_MANIFEST=/data/aishell_ner/test_aligned.jsonl
export CUDA_VISIBLE_DEVICES=0

INSTALL_DEPS=1 bash run_aligner.sh all
```

`run_aligner.sh stage0` 检查输入、`qwen-asr==0.0.6` 和 CUDA；stage1 对 marker-free
transcript 做批量对齐，并为每个 gold mention 补充时间范围；stage2 校验 mention
覆盖率、时间范围及 PCM16 WAV。同一实体重复出现时按 AISHELL-NER 转换得到的
`occurrence_index` 区分，不再靠 first/last 猜测。trace 和报告默认写入
`outputs/glclap/alignment/`，不会进入在线检索或提交链路。

## 最小启动方式

```bash
export HKUST_WORD_FREQ=/data/hkust/word_freq.txt
export MAGICDATA_WORD_FREQ=/data/magicdata/word_freq.txt
export AISHELL1_TRAIN_MANIFEST=/data/manifests/aishell1_train.jsonl
export AISHELL_NER_ANNOTATED_TRANSCRIPT=/data/AISHELL-NER/data/aishell_ner_transcript.test.txt
export AISHELL_NER_WAV_ROOT=/data/AISHELL-1/wav/test
export QWEN_MODEL=Qwen/Qwen3-ASR-0.6B
export CUDA_VISIBLE_DEVICES=0

# 先构建训练负词池并转换现成的 AISHELL-NER gold 标签。
bash run.sh stage0 stage1

# 再为边界实验生成离线参考时间戳。
INSTALL_DEPS=1 bash run_aligner.sh all

# 最后执行边界样本、训练、索引、检索、评测和测试。
bash run.sh stage2 stage3 stage4 stage5 stage6 stage7 stage8 stage9
```

## 阶段定义

| 阶段 | 作用 | 主要输出 |
|---|---|---|
| stage0 | Python、固定 Qwen/vLLM 版本、CUDA、源码和环境快照检查 | `outputs/glclap/run_metadata/` |
| stage1 | 训练负词池（1a）、AISHELL-NER gold 转换（1b）、评测库（1c） | catalog、两个 manifest 与报告 |
| stage2 | 消费独立 aligner 产物，前置静音生成七种 2 秒边界样本 | `test_boundary.jsonl` 和 boundary WAV |
| stage3 | 训练 global-only CLAP 基线 | `outputs/glclap/global_only/last.pt` |
| stage4 | 训练冻结原 Qwen projector 的主系统 | `outputs/glclap/frozen/last.pt` |
| stage5 | 训练 projector warm-start 与 random projector 消融 | `warmstart/last.pt`、`random/last.pt` |
| stage6 | 为每个 checkpoint 编码目标和 aliases，构建 fp16 精确索引 | `aishell_ner_10000.npz` 和元数据 |
| stage7 | 以 100 ms feed、2 秒 refresh 做累计音频 Top-50 检索 | `boundary_retrieval.jsonl` 和逐 chunk traces |
| stage8 | 计算 Recall/MRR、BoundaryPenalty、延迟、offline parity 和 bootstrap CI | 每个变体的 `metrics.json` |
| stage9 | 全量单测、100k catalog 与 10k×512 检索压力测试 | 测试日志和 artifact 列表 |

## 断点和消融控制

训练脚本每个 epoch 写 `epoch-XX.pt` 和 `last.pt`。默认
`RESUME_TRAINING=1`，再次运行 stage3/4/5 时会从对应 `last.pt` 恢复：

```bash
bash run.sh stage4
```

常用开关：

```bash
# 只跑 global-only 与 frozen 主系统
RUN_ABLATIONS=0 bash run.sh stage3 stage4 stage6 stage7 stage8

# 解码时不额外做整句 offline 等价性复算
VERIFY_OFFLINE=0 bash run.sh stage7

# stage9 跳过两个大规模压力烟测
RUN_STRESS=0 bash run.sh stage9

# 使用本地 Qwen 权重和其他输出根目录
QWEN_MODEL=/models/Qwen3-ASR-0.6B \
OUTPUT_ROOT=/exp/glclap_seed42 \
bash run.sh stage3 stage4
```

如果修改了输入数据、catalog 大小、chunk 大小或随机种子，应使用新的
`OUTPUT_ROOT`，不要在旧实验目录上混跑。`CHUNK_MS` 与 `CHUNK_SIZE_SEC` 必须
表达同一个时长，stage0 会检查二者一致。

## 多 GPU 使用

当前训练入口不是 DDP。推荐为不同初始化或 seed 启动独立实验目录：

```bash
CUDA_VISIBLE_DEVICES=0 OUTPUT_ROOT=/exp/seed42 SEED=42 RUN_ABLATIONS=0 bash run.sh stage3 stage4
CUDA_VISIBLE_DEVICES=1 OUTPUT_ROOT=/exp/seed43 SEED=43 RUN_ABLATIONS=0 bash run.sh stage3 stage4
```

不要让两个进程同时写同一个 `OUTPUT_ROOT`。完成训练后分别运行各自的
stage6–stage8，并保留 `run_metadata`、catalog 构建报告、配置快照和日志。
