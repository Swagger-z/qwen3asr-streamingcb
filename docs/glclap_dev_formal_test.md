# DEV 固定词表正式检索测试

`run_dev_test.sh` 使用与 test 相同的建库、检索和评测入口，独立测试已有
checkpoint，不训练模型、不运行 aligner、不生成边界音频。默认保留每句话的
全部 AISHELL-NER gold entity ID，输出离线和累计音频流式检索结果。

## 直接运行

在 Linux CUDA 仓库根目录，激活安装了项目依赖的环境后运行：

```bash
CUDA_VISIBLE_DEVICES=4 \
EXP_DIR=/data/zhengjie/research/qwen3asr-posttraining/qwen3asr-streamingcb/outputs/glclap/frozen_bs8_100epoch \
bash run_dev_test.sh all
```

默认顺序评测 `best.pt` 和 `last.pt`，每个 checkpoint 跑 offline/streaming 两种模式。
这是单进程推理，不会因设置多张可见 GPU 自动使用 DDP。
默认 CONFIG 为 `configs/glclap/qwen_post_projector_frozen.yaml`，加载 checkpoint
恢复已训练权重；该 YAML 的训练 epoch/batch 参数不会触发重新训练。

先查看命令而不写文件、不要求模型或数据存在：

```bash
DRY_RUN=1 bash run_dev_test.sh all
```

只测 best、只测 offline，或者更换结果目录：

```bash
CUDA_VISIBLE_DEVICES=4 CHECKPOINTS=best MODES=offline \
RUN_DIR=/data/zhengjie/research/qwen3asr-posttraining/qwen3asr-streamingcb/outputs/glclap/frozen_bs8_100epoch/dev_formal_offline \
bash run_dev_test.sh all
```

已存在的输出默认拒绝覆盖。重复推理建议指定新的 RUN_DIR，或在确认后设置
`OVERWRITE=1`。覆盖不会删除旧 trace 中已不属于新 manifest 的条目；以当前结果
JSONL 为准，换数据时使用新的 RUN_DIR。所有路径、模型和频次文件均可通过环境变量覆盖。

## 每个 stage 的输入和输出

默认 `RUN_DIR=${EXP_DIR}/dev_formal`。stage 编号属于这个新脚本，**不是 run.sh 的 stage 编号**。

| Stage | 输入及格式 | 输出/依赖 |
|---|---|---|
| stage0 | 独立原始文件：`DEV_ANNOTATED_TRANSCRIPT`，每行 `UTT_ID 带[PER]/(LOC)/<ORG>标注的文本`；`DEV_WAV_ROOT` 下的 AISHELL dev WAV | `data/targets_dev.jsonl`、`data/dev.jsonl`、`data/dev_entities.jsonl`、准备报告 |
| stage1 | stage0 的 dev target catalog、完整 dev manifest；独立 HKUST/MagicData `word_freq.txt`，每行 `词 频次` | `data/aishell_ner_dev_10000.jsonl`，固定 10k 目标+干扰词；`data/catalog_report.json` |
| stage2 | stage1 固定 dev 词表；独立训练结果 `${EXP_DIR}/{best,last}.pt`；Qwen 权重 | `{best,last}/aishell_ner_dev_10000.npz` 和对应 `.npz.json` 元数据 |
| stage3 | stage0 的实体句子 manifest；stage2 的每个索引及其对应 checkpoint | `{best,last}/{offline,streaming}.jsonl`、同名 `.jsonl.run.json` 配置/版本/输入哈希快照、`traces_{mode}/` |
| stage4 | stage3 的各模式结果 | `{best,last}/{offline,streaming}_metrics.json` |

stage0 默认从以下已有服务器路径读取数据（不修改这些原始输入）：

```text
/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/AISHELL-NER/data/aishell_ner_transcript.dev.txt
/data/zhengjie/datasets/asr/aishell/data_aishell/wav/dev
```

`DEV_MANIFEST` 示例，`source` 必须指向原始 dev WAV，`target_hotword_ids` 保留整句全部实体：

```json
{"key":"dev-001","source":"/data/aishell/wav/dev/dev-001.wav","target":"北京大学和清华大学","target_hotword_ids":["org-a","org-b"],"entities":[{"hotword_id":"org-a","text":"北京大学"},{"hotword_id":"org-b","text":"清华大学"}]}
```

这两个 ID 必须来自 `DEV_TARGET_CATALOG`，不要手动换成测试集 ID。
target catalog 示例（真实 ID 由 stage0 生成）：

```json
{"id":"org-a","text":"北京大学","aliases":[],"language":"zh","weight":1.0}
```

`DEV_FULL_MANIFEST` 包含完整 dev 数据，可含无实体句子；stage1 用它过滤干扰词，
与当前 test catalog builder 的做法一致。`DEV_MANIFEST` 只含有实体的句子，用于召回评测。
stage3 在加载模型前检查非空 gold、重复 key、100% 目标 ID 覆盖率，以及 checkpoint/catalog
哈希是否与索引元数据一致，防止混用 best/last 或误用 test 索引。

## 已准备过 dev 标签时

可以直接复用训练验证使用的原始 dev manifests，跳过 stage0：

```bash
CUDA_VISIBLE_DEVICES=4 \
EXP_DIR=/data/zhengjie/research/qwen3asr-posttraining/qwen3asr-streamingcb/outputs/glclap/frozen_bs8_100epoch \
DEV_TARGET_CATALOG=/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/AISHELL-NER/targets_dev.jsonl \
DEV_FULL_MANIFEST=/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/AISHELL-NER/dev.jsonl \
DEV_MANIFEST=/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/AISHELL-NER/dev_entities.jsonl \
bash run_dev_test.sh stage1 stage2 stage3 stage4
```

分阶段执行时，后续命令需要保留相同的环境变量。只重算指标可执行
`OVERWRITE=1 bash run_dev_test.sh stage4`；它不加载模型。

## 离线、流式与指标口径

- `--mode offline`：整句音频只编码/检索一次，忽略 refresh/feed 对计算分段的影响。
- `--mode streaming`：默认每 100 ms 喂音频，每 2 s 重编码全部累计音频；结束时处理尾块。
  不模拟真实时钟等待，也不是仅编码最新 2 s 的增量 encoder。
- 流式默认加 `--verify-offline`，额外进行一次整句检索并记录最终命中列表是否完全一致。
  `VERIFY_OFFLINE=0` 可关闭；校验额外开销不计入本次推理 elapsed/peak_memory。
- `hit_at_K`：本句任意一个正确实体进入前 K 的成功率。它与训练日志旧称
  `recall_at_K` 的含义接近，但候选规模和同分处理不同，仍不保证数值一致。
- `recall_at_K`：每句命中实体数/本句 gold ID 数，再按句平均，沿用 test evaluator。
  `evaluable_utterances` 和 `target_entity_count` 记录分母；重复 ID 去重。
- 例如三个目标只检出一个：Hit@1=1，Recall@1=1/3，Precision@1=1。
- `mrr` 基于返回的 Top-K 列表；默认 Top-50 外的目标贡献为 0。
- DEV 没有合成边界标签，boundary 指标不适用；未给 global-only baseline，
  `go_no_go_recall=false` 表示缺少对照，不能直接当作实验失败。offline 单独输出没有
  parity 标记，`go_no_go_offline_parity=false` 同样不是不一致；看 streaming 的 parity。
- offline 的 `first_top50_chunk_mean=0` 不表示零延迟，它只有一个编号为 0 的最终 batch。

这套流程让 dev 也执行固定 10k 全词表检索，但 dev/test 的词表内容并非完全相同。
为了最终横向比较，test 也应使用未按 mention/边界展开的 `test_entities.jsonl`，保留
整句全部 gold ID；边界展开集单独报告 focus-entity 指标。不能继续将 dev 的整句 Hit@K
与 test 的单目标 Recall@K 当作同一个指标。

脚本不修改训练验证函数、不重选 best.pt、不修补边界构建器；现有 checkpoint 可直接复用。
