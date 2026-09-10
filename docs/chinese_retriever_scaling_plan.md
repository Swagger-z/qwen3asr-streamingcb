# 中文检索器扩容实验计划（2026-09-07）

> **状态：已被多语言方案扩展。** 中文 C0/C1 仍是必要对照，但当前实现已经加入
> LibriSpeech 960、英文词级正例、STOP1/STOP2 和中英混合 catalog。可执行接口与
> 阶段说明见 [多语言检索器运行手册](multilingual_retriever_runbook.md)。

## 1. 实验结论先行

第一轮中文扩容训练使用以下音频训练集：

- AISHELL-1 train；
- AISHELL-2 train；
- MagicData train；
- HKUST train。

这里假设现有 MagicData/HKUST JSONL 都是训练 split，且字段语义与 AISHELL-1 的
`key/source/target` 一致。AISHELL-2 尚无 JSONL，需要新增转换和审计步骤。

第一轮只回答一个问题：**在模型、损失、负词库、dev/test 和训练更新数都固定时，
扩大中文训练音频池能否提升 AISHELL-NER 实体检索？**

因此第一轮继续使用：

- AISHELL-NER dev entity manifest 选择 `best.pt`；
- 固定 AISHELL-NER 10k catalog；
- AISHELL-NER original/boundary test 做离线和流式最终评测；
- 现有中文 2--8 字 local-positive 采样和 4,095 shared negatives。

MagicData/HKUST/AISHELL-2 的跨域 test 暂不进入 checkpoint 选择。普通 ASR JSONL
没有人工实体标注，后续要单独建立冻结的合成热词或人工标注评测协议。

## 2. 为什么不能只拼接四个 JSONL 后直接跑

当前训练入口能读取一个完整 JSONL，并在内存中加载、复制、shuffle 和分 batch。
直接拼接会留下四个问题：

1. 四个语料可能出现重复 `key`，checkpoint、trace 和缓存 provenance 不再可靠；
2. `epochs=10` 会让混合训练的 optimizer updates 远多于 AISHELL-1 baseline，无法判断
   收益来自数据多样性还是额外计算；
3. 大语料会按样本数自然主导训练，但日志没有记录每个 epoch 的实际语料构成；
4. 训练 checkpoint 目前只固定 dev 协议，没有固定训练 manifest 和采样协议，错误 resume
   不能被可靠拒绝。

四个语料均为中文，所以本轮**不需要**修改 2--8 字采样为英文词级采样；多语种采样
留到加入 LibriSpeech 时再做。

## 3. 统一训练 manifest 契约

派生后的混合 manifest 每行至少包含：

```json
{"key":"aishell2:ID001","source":"/data/aishell2/wav/ID001.wav","target":"中文转写","corpus":"aishell2","language":"zh","split":"train","source_key":"ID001"}
```

约束：

- `key` 必须是 `<corpus>:<source_key>`，跨语料全局唯一；
- `source` 必须能读取，训练前检查 WAV header 和采样率；
- `target` 必须非空，保留原始中文转写，不在数据准备阶段截取热词；
- `corpus` 取 `aishell1/aishell2/magicdata/hkust`；
- `language=zh`、`split=train`；
- 输入、输出 manifest 都记录 SHA-256、记录数和过滤原因。

重复 transcript 是正常现象，不能据此删除；重复 `source`、重复 namespaced key、空文本和
无法解析的音频应报错。可选的音频内容 hash 去重只做审计，不默认删除样本。

## 4. 需要修改或新增的代码

### P0：启动实验前必须完成

| 文件 | 变更 | 验收点 |
| --- | --- | --- |
| `scripts/prepare_aishell2_manifest.py`（新增） | 解析 AISHELL-2 transcript；递归扫描一个或多个 WAV root；按 stem 配对；输出带 `corpus/language/split/source_key` 的规范 JSONL 和报告 | 缺失音频、重复 stem、空文本、非预期采样率会明确报错；同一输入和 seed 输出 hash 一致 |
| `scripts/build_glclap_training_mix.py`（新增） | 接收重复的 `--input corpus=path`；规范化并合并 AISHELL-1/2、MagicData、HKUST；给 key 加语料 namespace；输出混合 manifest 与 provenance report | 报告四个语料的输入 hash、输入/输出数、过滤数、重复 key/source 和最终 hash |
| `asr/data/manifest.py` | 增加可选的 `manifest_corpus/language/split` 访问器及训练 manifest 严格校验；保留现有三字段接口兼容性 | 老 JSONL 仍可读；混合训练模式缺少或错误 metadata 时快速失败 |
| `asr/contextual/glclap_sampling.py`（新增） | 实现按 corpus 的确定性 epoch sampler；支持 `proportional` 和配置化权重；支持 `samples_per_epoch`；输出每个语料的计划/实际抽样数 | 相同 seed/epoch/world size 得到相同全局序列；各 rank 合并后覆盖同一全局计划；不足时的重复策略明确 |
| `scripts/train_glclap_retriever.py` | 使用 corpus-aware sampler 代替无条件全量 shuffle；按 `samples_per_epoch` 计算 scheduler/update 数；训练日志写逐语料样本数 | 单 AISHELL-1 默认配置与旧 shuffle 语义一致；混合训练可做等更新数比较 |
| `scripts/train_glclap_retriever.py` | 新增 `training_protocol`：训练 manifest/report hash、各语料记录数、采样策略、epoch size、负词库 hash；写入 config snapshot、log 和 checkpoint；resume 时严格比较 | 数据或采样协议变化后，旧 `last.pt` 必须拒绝续训 |
| `configs/glclap/chinese_scale.yaml`（新增） | 固定中文混合实验的 sampler、epoch size、seed、batch、负例和验证配置 | resolved config 能完整复现实验，不依赖未记录的命令行默认值 |
| `run_chinese_scale.sh`（新增） | 独立串联 AISHELL-2 转换、四语料审计/合并、global-only/frozen 训练、建库及原有 dev/test 评测 | 不改变当前 `run.sh` 的 AISHELL-1 论文流水线；每阶段可重跑和复用产物 |

### P1：根据数据规模决定是否随 P0 一起做

| 文件 | 变更 | 触发条件 |
| --- | --- | --- |
| `asr/data/manifest_dataset.py` | 把占位实现替换为带 byte-offset index 的只读 JSONL dataset，按采样 index 延迟读取记录 | 四语料 JSONL 全量加载导致启动慢或 host 内存不可接受 |
| `scripts/train_glclap_retriever.py` | 不再构造完整 `shuffled` 和 `epoch_batches` list，改为 index iterator，并用已知 batch 数判断 accumulation 尾批 | manifest 达到百万级记录，Python list/dict 副本成为明显开销 |

建议先用合并报告记录总行数、JSONL 大小和一次加载的峰值内存。若峰值达到预先分配给
数据加载器的 host 内存预算，直接完成 P1，不要等正式训练 OOM 后再改。

## 5. 测试改动

需要新增：

- `tests/test_prepare_aishell2_manifest.py`：正常配对、多个 WAV root、缺失/重复 stem、
  空 transcript、输出稳定性；
- `tests/test_glclap_training_mix.py`：四语料合并、key namespace、字段补全、重复 source、
  输入/输出 hash 和报告计数；
- `tests/test_glclap_sampling.py`：proportional、显式权重、`samples_per_epoch`、多 epoch
  确定性、DDP 分片和小语料重复；
- 扩展 `tests/test_manifest_schema.py`：旧三字段兼容和混合训练严格模式；
- 扩展 `tests/test_glclap_negative_filtering.py`：混合 batch 中所有已说 2--8 字片段仍不会
  被采为 shared negative；
- `tests/test_run_chinese_scale_contract.py`：runner 固定四个训练输入、独立 dev/test、
  global-only/frozen 配对和新的 output root。

P0 完成后先运行 dependency-free 全量测试：

```bash
python -m unittest discover -s tests -v
```

Linux CUDA 上再做一个每语料少量样本的 smoke run，确认 packed audio、feature cache、DDP、
validation 和 resume protocol 都工作，再提交完整训练。

## 6. 第一轮实验矩阵

先冻结以下公共条件：Qwen checkpoint、adapter 初始化、loss 权重、global batch 384、
negative catalog、AISHELL-NER dev/test、10k evaluation catalog、seed、2 秒 refresh 和
评测代码 commit。

| 实验 | 训练池 | 每 epoch 样本数 | sampler | 作用 |
| --- | --- | ---: | --- | --- |
| C0-global | AISHELL-1 | `N_aishell1` | 旧语义/proportional | 数据扩容的 global-only 基线 |
| C0-frozen | AISHELL-1 | `N_aishell1` | 旧语义/proportional | 当前主系统可复现基线 |
| C1-global | 四语料中文池 | `N_aishell1` | proportional | 等计算量下检验训练池多样性 |
| C1-frozen | 四语料中文池 | `N_aishell1` | proportional | 第一轮主要比较对象 |

如果现有 C0 checkpoint 的代码、训练标签修复、dev protocol、negative catalog 和更新数
hash 完全一致，可以复用；只要有一项不同就重跑 C0。

C1 仍训练 10 个逻辑 epoch，但每个 epoch 只确定性抽取 `N_aishell1` 条，因此 C0/C1 的
optimizer updates、warmup 和 cosine scheduler 完全一致。第一轮先使用 proportional，
不同时引入 corpus balance 超参。

若 C1 有收益，再追加：

- C2：四语料、固定相同总 updates，使用 `p_i ∝ n_i^0.5` 的 temperature sampling；
- C3：四语料 full-data pass，评估增加训练计算后的上限；
- leave-one-corpus-out：判断收益来自 AISHELL-2、MagicData 还是 HKUST。

## 7. 评测与决策门槛

checkpoint 只按 AISHELL-NER dev Recall@50 选择。最终在同一冻结 test 上报告：

- 离线 Recall@1/5/10/20/50、MRR；
- original fast/realtime 的 first-complete-refresh Recall@50；
- deadline Recall@50（0/100/200/500/1000/2000 ms）；
- miss/dropout、成功检出 latency P50/P95；
- boundary penalty 与配对 bootstrap CI；
- search/processing P50/P95、RTF 和 peak memory。

第一轮的判断顺序：

1. C1-frozen 相对 C0-frozen 的 AISHELL-NER dev/test 是否稳定改善，并报告同一评测样本上的
   配对 bootstrap 差值与置信区间；
2. C1-frozen 相对 C1-global 的 Recall@50 配对增益是否达到现有 `>=5pp` 且 CI 下界大于 0；
3. boundary penalty 是否 `<=0.03`；
4. 离线最终检索 parity 是否 100%；
5. search P95 是否 `<10 ms`，若失败则单独标为工程性能问题，不与召回结论混写。

如果 C1 在等计算量下退化，先检查实际语料占比、坏音频/文本、AISHELL-1 暴露量和域偏移，
再运行 C2；不要直接通过增加 epoch 掩盖问题。

## 8. 本轮明确不做

- 不加入 LibriSpeech，不修改英文正例采样；
- 不改 prompt、token bias、候选生命周期或 commit/rollback；
- 不用 test 选择 checkpoint；
- 不把 MagicData/HKUST word-frequency list 当 gold 热词标注；
- 不同时修改 negative catalog 大小、chunk 大小或检索结构。

这样得到的 C0/C1 差异可以主要归因于中文训练音频池的扩大，而不是其他系统变量。
