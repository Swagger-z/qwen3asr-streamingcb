# GLCLAP 完整分阶段实验运行手册

仓库根目录的 `run.sh` 串联了 catalog 构建、边界数据生成、四种模型训练、
索引构建、累计音频检索、评测和压力测试。脚本面向 Linux CUDA 单机，训练
支持单卡和 torchrun DDP；多张 GPU 既可用于同一实验，也可划分后并行运行不同实验。

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
| `AISHELL_NER_DEV_ANNOTATED_TRANSCRIPT` | `data/raw/AISHELL-NER/data/aishell_ner_transcript.dev.txt` | 官方 dev `UTT_ID TAGGED_TRANSCRIPT` |
| `AISHELL_NER_DEV_WAV_ROOT` | `data/raw/AISHELL-1/wav/dev` | 可递归扫描的 dev WAV 根目录 |
| `AISHELL_NER_ANNOTATED_TRANSCRIPT` | `data/raw/AISHELL-NER/data/aishell_ner_transcript.test.txt` | 官方 `UTT_ID TAGGED_TRANSCRIPT` |
| `AISHELL_NER_WAV_ROOT` | `data/raw/AISHELL-1/wav/test` | 可递归扫描的 test WAV 根目录 |

stage1b 解析 AISHELL-NER dev，供 checkpoint validation；stage1c 独立解析 test，
供最终论文评测。两步都只读取已有的 `[PER]`、`(LOC)`、`<ORG>` gold marker，
不会根据热词表推断实体。

| dev 派生变量 | 默认路径 | 消费者 |
|---|---|---|
| `AISHELL_NER_DEV_TARGET_CATALOG` | `data/aishell_ner/targets_dev.jsonl` | 审计 dev gold 实体 |
| `AISHELL_NER_DEV_EVAL_MANIFEST` | `data/aishell_ner/dev.jsonl` | 全量 dev，包含无实体句 |
| `AISHELL_NER_DEV_ENTITY_MANIFEST` | `data/aishell_ner/dev_entities.jsonl` | stage3--5 validation |

| test 派生变量 | 默认路径 | 消费者 |
|---|---|---|
| `AISHELL_NER_TARGET_CATALOG` | `data/aishell_ner/targets_test.jsonl` | stage1d、aligner、建库 |
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
export AISHELL_NER_DEV_ANNOTATED_TRANSCRIPT=/data/AISHELL-NER/data/aishell_ner_transcript.dev.txt
export AISHELL_NER_DEV_WAV_ROOT=/data/AISHELL-1/wav/dev
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
| stage1 | 训练负词池（1a）、AISHELL-NER dev/test gold 转换（1b/1c）、测试库（1d） | dev/test catalog、manifest 与报告 |
| stage2 | 消费独立 aligner 产物，前置静音生成七种 2 秒边界样本 | `test_boundary.jsonl` 和 boundary WAV |
| stage3 | 训练并验证 global-only CLAP 基线 | `outputs/glclap/global_only/{last,best}.pt` |
| stage4 | 训练并验证冻结原 Qwen projector 的主系统 | `outputs/glclap/frozen/{last,best}.pt` |
| stage5 | 训练并验证 projector 初始化消融 | `warmstart/{last,best}.pt`、`random/{last,best}.pt` |
| stage6 | 为每个 checkpoint 编码目标和 aliases，构建 fp16 精确索引 | `aishell_ner_10000.npz` 和元数据 |
| stage7 | 以 100 ms feed、2 秒 refresh 做累计音频 Top-50 检索 | `boundary_retrieval.jsonl` 和逐 chunk traces |
| stage8 | 计算 Recall/MRR、BoundaryPenalty、延迟、offline parity 和 bootstrap CI | 每个变体的 `metrics.json` |
| stage9 | 全量单测、100k catalog 与 10k×512 检索压力测试 | 测试日志和 artifact 列表 |

## 训练期验证

stage3--5 强制使用 stage1b 生成的 `AISHELL_NER_DEV_ENTITY_MANIFEST`。每条语音的
local positives 直接来自 AISHELL-NER dev 的 `entities[].text`，不再从 transcript
随机截取；同一句中的多个不同 gold 实体会共同标为正样本。训练入口拒绝没有实体标注
的验证记录，并检查其 `key` 不与 AISHELL-1 train 重叠。AISHELL-NER test 始终隔离，
只供 stage6--8 最终评测，不参与 `best.pt` 选择。

默认每个 epoch 结束验证一次：

```bash
EVAL_STRATEGY=epoch EVAL_MAX_SAMPLES=0 bash run.sh stage4
```

也可按固定 optimizer update 数验证；这里的 step 是完成 gradient accumulation 后的
`optimizer.step()`，不是 micro batch：

```bash
EVAL_STRATEGY=steps EVAL_STEPS=100 EVAL_MAX_SAMPLES=512 bash run.sh stage4
```

`EVAL_MAX_SAMPLES=0` 默认使用完整 dev。固定 step 验证成本较高，可显式设为 512，
此时会用固定 seed 选择同一个 dev 子集，保证各次验证可比。`EVAL_BATCH_SIZE`
默认 8。每次验证把 validation loss、
global/local loss、Recall@1/5/10/20/50 和 MRR 追加到 `train.jsonl`。默认按
`Recall@50` 保存 `best.pt`；stage6/7 消费 `best.pt`，而 `last.pt` 只用于断点恢复。
`steps` 模式若训练结束时未正好命中间隔，会额外做一次 final validation，保证产生
`best.pt`。

验证集和调度在进程启动时读取；给已有运行中的训练修改环境变量不会生效，必须
停止后从新实验目录重启。

## 断点和消融控制

训练脚本每个 epoch 写 `epoch-XX.pt` 和 `last.pt`，并在验证指标改善时写
`best.pt`。best metric/epoch/global step 同步保存在 checkpoint，默认
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
`OUTPUT_ROOT`，不要在旧实验目录上混跑。checkpoint 会保存验证协议名、AISHELL-NER
dev manifest SHA-256、验证 seed 和子集大小；任一项不一致都会拒绝续训，防止把旧的
随机子串 best Recall 与新的 gold-entity Recall 混合比较。`CHUNK_MS` 与 `CHUNK_SIZE_SEC` 必须
表达同一个时长，stage0 会检查二者一致。

## 训练性能与缓存

stage3--5 默认启用以下训练热路径优化：

- `AUDIO_BATCHING=packed`：同一 micro batch 的变长特征按有效长度拼接，一次调用
  Qwen `audio_tower`，随后按官方卷积输出长度拆回各 utterance。设置为 `serial`
  可恢复官方逐音频路径，用于精度对照；packed/serial 使用隔离的缓存 namespace。
- `AUDIO_LOADER_WORKERS=4`：持久线程并行读取 WAV；PCM16 转 float32 使用 NumPy
  `frombuffer`，不再逐 sample 执行 Python `int.from_bytes`。
- `FEATURE_CACHE_DIR`：把 frozen 模式需要的 post-projector `[T,1024]` 或
  warmstart/random 需要的 pre-projector `[T,896]` 写到分片磁盘缓存。首轮 miss
  仍需运行 AuT，后续 epoch 和相同 Qwen/模式的实验直接读取缓存。
- `TEXT_CACHE_MAX_ENTRIES=20000`：在每个 rank 的设备上缓存冻结 token embedding
  mean pooling 结果；主系统的 4095 个负词仍会经过可训练文本 MLP，实验目标没有改变。
- global-only 配置的 `local_weight=0` 在训练时直接跳过局部正负样本、hotword MLP
  和 `B×T×K` logits；验证仍计算完整局部检索矩阵与 Recall@K，评估口径不变。
- 训练损失在设备上累计到 epoch 末统一读取，避免每个 micro batch 因
  `float(loss)` 触发三次 CUDA 同步；延迟测量用的 `cuda.synchronize()` 只保留在
  显式推理 benchmark 路径。

建议把 `FEATURE_CACHE_DIR` 放在所有训练 rank 可见的高速本地 NVMe 或共享 SSD：

```bash
FEATURE_CACHE_DIR=/fast/glclap_qwen_cache \
AUDIO_BATCHING=packed AUDIO_LOADER_WORKERS=8 \
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run.sh stage4
```

缓存 key 包含 Qwen 路径、固定 qwen-asr 版本、packed/serial 模式、音频绝对路径、
文件大小和修改时间。更换权重但复用同一路径时，应改用新的
`FEATURE_CACHE_DIR`。缓存只包含冻结特征，不包含数据集 WAV、训练 checkpoint
或 embedding index。

`train.jsonl` 的 `train_epoch` 事件额外记录：

- `epoch_seconds` 和全局 `samples_per_second`；
- host 侧 cache load、WAV I/O、encoder submit、cache write 与文本准备耗时；
- 音频/文本缓存 hit、miss、write 和 entry 数；
- 实际 `audio_batching` 模式。

host 分阶段耗时用于定位数据等待，不能相加当作 GPU kernel 时间；端到端吞吐以
`samples_per_second` 为准。第一次填充磁盘缓存时 cache write 包含一次合并后的
GPU→CPU 等待，因此通常明显慢于第二个 epoch。多个变体同时冷启动时可能重复计算
同一个 cache miss；要测稳定训练吞吐，应先让一个同 feature-kind 的变体完成首轮
缓存填充。

## 多 GPU 使用

stage3--5 已支持单机多卡 DDP。`run.sh` 默认从 `CUDA_VISIBLE_DEVICES` 的逗号
分隔项数推断 `NUM_GPUS`；显式设置 `NUM_GPUS` 可覆盖。以下命令会让一个 frozen
实验同时使用物理卡 4、5、6、7：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 RUN_ABLATIONS=0 bash run.sh stage4
```

内部等价 launcher 为：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_glclap_retriever.py ...
```

训练数据在每个 epoch 做一次全局确定性 shuffle，再像 `DistributedSampler` 一样
补齐并按 rank 等长分片。梯度累计期间使用 DDP `no_sync()`，只在 optimizer update
同步。验证集不补齐地分片到所有 rank，每条 dev 样本只评估一次，最后按样本数
all-reduce 聚合 loss、Recall 和 MRR；日志、`best.pt`、`last.pt` 和 epoch
checkpoint 仍只由 rank 0 写，避免文件竞争。

配置中的 `training.global_batch_size=384` 是所有 GPU 合计的 batch，不是每卡
batch。`micro_batch_size=8` 时自动得到：

| GPU 数 | 每卡 gradient accumulation | 全局有效 batch |
|---:|---:|---:|
| 1 | 48 | 384 |
| 4 | 12 | 384 |
| 8 | 6 | 384 |

`GLOBAL_BATCH_SIZE` 必须能被 `micro_batch_size × NUM_GPUS` 整除，否则训练在加载
模型前报错。只有把 `training.global_batch_size` 设为 0，才退回旧的“每卡固定
`gradient_accumulation_steps`”语义；这种情况下 GPU 数增加会同时放大全局 batch。

`bash run.sh stage3 stage4 stage5` 仍按变体顺序执行，但每个变体都会使用全部
`NUM_GPUS`。如果要同时跑多个变体，应把 GPU 划成互不重叠的组，并使用不同的
`OUTPUT_ROOT`，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 OUTPUT_ROOT=/exp/global bash run.sh stage3 &
CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 OUTPUT_ROOT=/exp/frozen bash run.sh stage4 &
wait
```

每个 rank 都会各自加载一份冻结 Qwen AuT，因此显存不会跨卡共享。正式环境使用
NCCL；stage0 会检查可见 GPU 数不少于 `NUM_GPUS`。
