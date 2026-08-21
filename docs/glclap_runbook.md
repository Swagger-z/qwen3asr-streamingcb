# GLCLAP 完整分阶段实验运行手册

仓库根目录的 `run.sh` 串联了 catalog 构建、边界数据生成、四种模型训练、
索引构建、累计音频检索、评测和压力测试。脚本面向 Linux CUDA 单机，训练
入口仍是单进程；多张 GPU 用于从不同 shell 并行运行不同实验。

## 外部必备输入

脚本不会下载或提交数据。运行前需要准备以下文件：

| 环境变量 | 默认路径 | 必需字段/格式 |
|---|---|---|
| `HKUST_WORD_FREQ` | `data/raw/hkust/word_freq.txt` | 每行 `词语 频次` |
| `MAGICDATA_WORD_FREQ` | `data/raw/magicdata/word_freq.txt` | 每行 `词语 频次` |
| `AISHELL1_TRAIN_MANIFEST` | `data/aishell1/train.jsonl` | `utt_id/audio/text` |
| `AISHELL1_NE_TARGET_CATALOG` | `data/aishell1_ne/targets_test.jsonl` | 标准 `id/text/aliases/...` catalog |
| `AISHELL1_NE_EVAL_MANIFEST` | `data/aishell1_ne/test.jsonl` | `utt_id/audio/text/target_hotword_ids` |
| `AISHELL1_NE_ALIGNED_MANIFEST` | `data/aishell1_ne/test_aligned.jsonl` | 上述字段加 `hotword_start_sec/hotword_end_sec` |

`AISHELL1_NE_ALIGNED_MANIFEST` 由离线 Qwen3 ForcedAligner 或其他参考对齐器
生成。在线检索和解码阶段不会调用 forced aligner。目标 catalog 与 manifest
中的 `target_hotword_ids` 必须一致；构建器会在 stage1 中验证。

## 最小启动方式

```bash
export HKUST_WORD_FREQ=/data/hkust/word_freq.txt
export MAGICDATA_WORD_FREQ=/data/magicdata/word_freq.txt
export AISHELL1_TRAIN_MANIFEST=/data/manifests/aishell1_train.jsonl
export AISHELL1_NE_TARGET_CATALOG=/data/manifests/aishell1_ne_targets.jsonl
export AISHELL1_NE_EVAL_MANIFEST=/data/manifests/aishell1_ne_test.jsonl
export AISHELL1_NE_ALIGNED_MANIFEST=/data/manifests/aishell1_ne_test_aligned.jsonl
export QWEN_MODEL=Qwen/Qwen3-ASR-0.6B
export CUDA_VISIBLE_DEVICES=0

# 首次运行时安装固定依赖；已有正确环境时保持 INSTALL_DEPS=0。
INSTALL_DEPS=1 bash run.sh stage0

# 完整顺序运行。
bash run.sh all
```

如果 stage0 已单独执行，完整实验可从 stage1 开始：

```bash
bash run.sh stage1 stage2 stage3 stage4 stage5 stage6 stage7 stage8 stage9
```

## 阶段定义

| 阶段 | 作用 | 主要输出 |
|---|---|---|
| stage0 | Python、固定 Qwen/vLLM 版本、CUDA、源码和环境快照检查 | `outputs/glclap/run_metadata/` |
| stage1 | 合并 HKUST/MagicData 词频并构建两套 10k catalog | `data/hotwords/*.jsonl` 与构建报告 |
| stage2 | 前置静音生成七种 2 秒 chunk 边界样本 | `test_boundary.jsonl` 和 boundary WAV |
| stage3 | 训练 global-only CLAP 基线 | `outputs/glclap/global_only/last.pt` |
| stage4 | 训练冻结原 Qwen projector 的主系统 | `outputs/glclap/frozen/last.pt` |
| stage5 | 训练 projector warm-start 与 random projector 消融 | `warmstart/last.pt`、`random/last.pt` |
| stage6 | 为每个 checkpoint 编码目标和 aliases，构建 fp16 精确索引 | `aishell1_ne_10000.npz` 和元数据 |
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
