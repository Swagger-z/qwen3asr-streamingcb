# GLCLAP 累计音频热词检索模块 v1

本模块是独立检索闭环，不接入 prompt、logit bias、CandidateManager 或
commit/rollback。现有 AC、phoneme-CTC retriever 及其导入路径不变。

## 架构

默认配置 `qwen_post_projector_frozen` 冻结 Qwen3-ASR-0.6B 的 AuT、原
projector 和 LLM token embedding：

```text
16 kHz PCM
  -> Qwen AuT ln_post [T,896]
  -> Qwen proj1/GELU/proj2 [T,1024] (冻结)
  -> audio MLP adapter [T,512]

hotword text
  -> Qwen token embedding mean pool [1024] (冻结)
  -> text MLP adapter [512]

score(h) = max_t dot(audio_t, text_h)
```

`QwenGLCLAPEncoder` 通过临时 forward hook 只读取得 `ln_post` 输出，官方
Qwen 源码和 checkpoint 不会被修改。checkpoint 只保存 retrieval branch，
不会重复保存 Qwen 权重。

三种初始化配置：

- `configs/glclap/qwen_post_projector_frozen.yaml`：论文默认方案。
- `configs/glclap/qwen_projector_warmstart.yaml`：复制 `proj1/proj2` 到独立
  retrieval branch，第 0 个 epoch 冻结，从第 1 个 epoch 起使用 `3e-5`。
- `configs/glclap/random_projector.yaml`：同形状随机初始化，并采用相同的
  冻结/学习率日程，保证初始化消融公平。
- `configs/glclap/global_only_clap.yaml`：`local_weight=0` 的 Go/No-Go 基线。

## 数据 schema

AISHELL-1 训练 manifest（JSONL）：

```json
{"key":"BAC009S0002W0122","source":"/data/aishell/wav.wav","target":"欢迎来到人工智能岛"}
```

每个 epoch 根据 `seed/epoch/utt_id` 确定性采样一个 2--8 字连续 local
positive。负词文件可以是版本化 hotword JSONL、每行一个中文词的 UTF-8
文本，也能读取 `词语 频次` 格式；正式实验应先用
`build_glclap_training_pool.py` 合并和过滤多个词频来源。每个 batch 共享采样 4095
个不同负词，并排除完整 transcript、采样到的 positive，以及 transcript 中
全部 2--8 字连续子串，防止真实说出的词成为假负例。

Dev/Test-AISHELL-NER manifest 在上述字段外增加：

```json
{"target_hotword_ids":["poi-0001"],"boundary_group":"Cross-50"}
```

hotword catalog 继续使用项目已有 schema：`id/text/aliases/language/weight/metadata`。
建库时 canonical text 和每个 alias 分别编码，检索时按 ID 取最高 variant 分数。
HKUST/MagicData `word_freq.txt` 只作为训练负词和评测 distractor 来源，不作为
gold 命名实体。评测 10k catalog 必须由 AISHELL-NER gold surface forms 加
distractor 组成；构建器会排除 gold targets 和评测 transcript 中已说出的候选词，
并验证 manifest 的 `target_hotword_ids`。完整 schema、泄漏约束和边界构建见
`docs/glclap_catalog_preparation.md`。

## 可复现命令

Linux CUDA 环境使用固定版本 `qwen-asr==0.0.6`、
`transformers==4.57.6`、`vllm==0.14.0`：

```bash
pip install -e '.[probe,qwen,config]'

python scripts/prepare_aishell_ner.py \
  --annotated-transcript /data/AISHELL-NER/data/aishell_ner_transcript.test.txt \
  --wav-root /data/AISHELL-1/wav/test \
  --target-catalog-output data/aishell_ner/targets_test.jsonl \
  --eval-manifest-output data/aishell_ner/test.jsonl \
  --entity-manifest-output data/aishell_ner/test_entities.jsonl \
  --report data/aishell_ner/test_preparation_report.json \
  --split test

python scripts/build_glclap_training_pool.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --output data/hotwords/zh_train_10k.jsonl \
  --report data/hotwords/training_pool_report.json \
  --size 10000 \
  --seed 42

python scripts/build_glclap_evaluation_catalog.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --target-catalog data/aishell_ner/targets_test.jsonl \
  --eval-manifest data/aishell_ner/test.jsonl \
  --output data/hotwords/aishell_ner_10k.jsonl \
  --report data/hotwords/evaluation_catalog_report.json \
  --size 10000 \
  --seed 42

python scripts/train_glclap_retriever.py \
  --config configs/glclap/qwen_post_projector_frozen.yaml \
  --manifest data/aishell1/train.jsonl \
  --dev-manifest data/aishell1/dev.jsonl \
  --negative-catalog data/hotwords/zh_train_10k.jsonl \
  --output-dir outputs/glclap/frozen

python scripts/build_glclap_index.py \
  --config configs/glclap/qwen_post_projector_frozen.yaml \
  --checkpoint outputs/glclap/frozen/best.pt \
  --catalog data/hotwords/aishell_ner_10k.jsonl \
  --output outputs/glclap/frozen/aishell_ner_10k.npz

python scripts/decode_streaming_retrieval.py \
  --config configs/glclap/qwen_post_projector_frozen.yaml \
  --checkpoint outputs/glclap/frozen/best.pt \
  --index outputs/glclap/frozen/aishell_ner_10k.npz \
  --manifest data/aishell_ner/test_boundary.jsonl \
  --output outputs/glclap/frozen/test.jsonl \
  --trace-dir outputs/glclap/frozen/traces \
  --verify-offline

python scripts/eval_hotword_retrieval.py \
  --input outputs/glclap/frozen/test.jsonl \
  --baseline outputs/glclap/global_only/test.jsonl \
  --output outputs/glclap/frozen/metrics.json
```

训练默认每卡 micro batch 为 8，全局有效 batch 为 384。单卡自动累计 48 步，
4 卡累计 12 步，8 卡累计 6 步。训练入口支持 `torchrun` 单机多卡 DDP：各 rank
独立加载冻结 Qwen、等长分片数据，并只在 optimizer update 同步梯度；rank 0
独占验证、日志和 checkpoint 写入。`training.global_batch_size` 必须能被
`micro_batch_size × WORLD_SIZE` 整除。`run.sh` 可直接使用
`CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run.sh stage4`。

训练必须提供独立 dev manifest，且其 `key` 不得与 train 重叠。默认
`evaluation.strategy=epoch`，每轮结束计算 validation loss、Recall@1/5/10/20/50
和 MRR，并按 Recall@50 保存 `best.pt`。若改为固定 optimizer step：

```bash
python scripts/train_glclap_retriever.py ... \
  --override evaluation.strategy=steps --override evaluation.steps=100
```

## 流式与索引约束

- `step(pcm16k)` 可以接收任意长度输入；每累计满 2 秒返回一次结果。
- 每次 refresh 都重新编码从 0 到当前边界的全部音频；`finish()` 对尾块再做
  一次完整累计编码。整 2 秒结束时复用最后一次结果而不重复编码。
- 一个 active session 不允许替换 index，必须先 `reset()`；各 session 的 PCM、
  chunk ID 和结果均为独立状态。
- key 以 fp16 存储；score 时分块转 fp32，默认 block size 16384。检索是精确
  矩阵乘法，不是 ANN。流式 CUDA 命令会在开始时把约 20 MB 的 10k×512
  fp32 scoring cache 放到与 adapter 相同的 GPU；NumPy 输入仍走 CPU GEMM，
  两条路径使用相同的 alias max 聚合和 Top-K 语义。
- 数据、Qwen 权重、retriever checkpoint 和 `.npz` index 不提交仓库。

## 指标和决策门槛

评测命令报告 Recall@1/5/10/20/50、MRR、Precision/F1/FAR、
BoundaryPenalty、首次进入 Top-50 的 chunk、各阶段 P50/P95、RTF、显存和
offline/final parity。这里 FAR 定义为 Top-K 中非目标项比例，即
`1 - Precision@K`。

Go/No-Go 条件：

1. 相比 global-only CLAP，Recall@50 至少提高 5 个百分点，配对 bootstrap
   95% CI 下界大于 0。
2. `BoundaryPenalty = Recall_center@50 - Recall_cross@50 <= 0.03`。
3. 使用 `--verify-offline` 时 final accumulated 与离线整句 Top-K 完全一致。
4. 10 秒音频、10k catalog 的 `search_ms` P95 小于 10 ms；AuT 累计重编码
   时间单独报告，不计入 score-only 门槛。

只有 warm-start 在 dev Recall@50 比冻结方案提高至少 2 个百分点且训练无
NaN/发散时，才把 warm-start 升级为后续主配置。
