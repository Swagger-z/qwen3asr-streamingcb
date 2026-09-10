# 实验进展与下一步计划（2026-09-03）

## 1. 当前判断

同步推进“检索器扩容”和“热词增强注入”是合理的，两条线分别回答：

1. 检索器能否从单一中文语料扩展到跨语料、跨语言并保持泛化；
2. 已检索到的热词能否安全地作用于 Qwen3-ASR，而不污染稳定前缀或扩大回滚。

两条线不需要互相等待：热词增强可以先使用 oracle 候选和固定的检索回放结果，
检索器扩容则可以在不改解码策略的情况下独立训练和评测。合流前必须固定候选接口、
数据划分和评测口径，避免同时更换训练数据和注入方法后无法归因。

## 2. 当前实验进展

当前已完成 GLCLAP 风格检索分支、固定 10k 索引、整句离线检索、累积音频流式检索、
fast FIFO 回放和在线时效指标。当前结果仍是“检索器评测”，没有覆盖完整的
prompt、ASR 解码、token bias 和 commit/rollback 链路。

本次记录基于 `best_original_fast_metrics.json` 和
`last_original_fast_metrics.json`：2,376 条评测音频、3,372 个目标实体、
7,682 次检索刷新，2 秒累积音频刷新，fast FIFO 模拟。

| 指标 | best | last |
| --- | ---: | ---: |
| Recall@1 | 30.33% | 33.43% |
| Recall@50 | 90.41% | 91.99% |
| MRR | 0.5425 | 0.5854 |
| first-complete-refresh Recall@50 | 91.01% | 92.65% |
| deadline Recall@50，1 秒 | 57.47% | 58.54% |
| deadline Recall@50，2 秒 | 91.10% | 92.67% |
| 成功样本检出延迟 P50 / P95 | 890 / 1,790 ms | 834 / 1,789 ms |
| 检索耗时 P95 | 14.99 ms | 14.84 ms |
| 完整检索处理 P95 | 32.91 ms | 32.60 ms |
| 离线最终检索结果一致率 | 100% | 100% |

`last` 在这次 test 结果上整体优于 `best`，但这只能作为观察，不能据此重新选择
checkpoint；正式选择仍必须只看固定 dev 集，test 只做一次最终报告。

当前尚未闭环：

- `go_no_go_recall=false`：缺少配对的 global-only 基线，尚不能判断局部目标收益是否过门限；
- `go_no_go_boundary=false`：当前 original 集没有 center/cross 边界配对样本，不能把
  `boundary_penalty=0` 当成已通过；
- 检索耗时 P95 约 14.8 ms，尚未达到现有 `<10 ms` 门限；
- 当前只有 fast FIFO 结果，还需要 real-time 回放；
- 2 秒刷新使 deadline recall 强依赖 chunk 相位，不能只把它解释为模型检索能力；
- GLCLAP 的 `RetrievalBatch` 目前还没有适配到主流式会话的 `RetrievalUpdate` 和
  `CandidateManager`。

因此，当前可以认为“累积音频流式检索原型和指标链路已经跑通”，但不能认为
“流式热词增强任务已经完成”。

## 3. 研究线 A：扩大检索器训练

### 3.1 数据范围

计划训练集：

- 中文：AISHELL-1 train、AISHELL-2 train、MagicData train、HKUST train；
- 英文：LibriSpeech train-clean-100、train-clean-360、train-other-500。

建议 dev 集必须独立补齐：

- 中文主选择：AISHELL-NER dev；
- 中文跨域监控：MagicData dev，并保留 AISHELL-2 dev 作为训练域监控；
- 英文：LibriSpeech dev-clean 和 dev-other。

建议最终 test：

- 中文：AISHELL-NER test、MagicData test；
- 英文：LibriSpeech test-clean 和 test-other。

普通 AISHELL-1、MagicData 和 LibriSpeech ASR test transcript 本身不是热词标注。
如果要在这些 split 上报告检索 Recall@K，必须预先固定、版本化并冻结目标热词抽取、
负例 catalog 和去泄漏协议；否则不同实验的 Recall@K 不可比。中文实体主结论优先使用
AISHELL-NER 的人工标注，普通 transcript 上的自动抽样结果应标为合成热词任务。

### 3.2 扩容前必须修改的训练协议

1. **多语种正例采样**：当前实现会删除空格后随机截取 2–8 个字符，适合当前中文小实验，
   不适合 LibriSpeech。中文和英文应分别使用明确的采样单元；英文至少按词或 tokenizer
   单元取连续短语，不能产生跨词拼接的字符碎片。
2. **同语言负例**：中英混合时不能主要依靠跨语言的简单负例。batch 内负例和 4,095 个
   shared negatives 应按语言分池，并记录 hard-negative 策略。
3. **语料平衡**：当前单 manifest 均匀 shuffle 会让大语料主导训练。需要显式的
   corpus/language sampler，报告每个 epoch 实际看到的各语料样本数。
4. **统一 schema**：每条数据增加 `corpus`、`language`、`split`，`key` 使用语料前缀，
   音频统一为 16 kHz，并检查跨语料文本、音频和 speaker 泄漏。
5. **多 dev 选择**：训练入口当前只接受一个 dev manifest，且以单一 Recall@50 选 best。
   应支持逐域指标和跨域宏平均，并设置单域退化 guardrail；禁止用 test 选择 checkpoint。
6. **冻结特征缓存**：Qwen 声学塔冻结，扩容训练应优先缓存声学特征；缓存命名空间必须包含
   模型版本、特征位置和唯一语料 key。

### 3.3 推荐的分阶段实验

| 阶段 | 训练数据 | 目的 |
| --- | --- | --- |
| A0 | AISHELL-1 | 固定现有可复现实验和多 dev 评测基线 |
| A1 | AISHELL-1 + AISHELL-2 + MagicData + HKUST | 验证中文扩容收益，先排除多语种采样变量 |
| A2 | A1 + LibriSpeech 960h | 验证双语共享空间和跨语言干扰 |
| A3 | A2 的采样/负例消融 | 区分数据量、语言配比、同语言负例各自贡献 |

所有阶段都报告逐语料结果和等权宏平均，不能只看拼接后的总体 micro 指标。核心指标保留
Recall@1/10/50、MRR、first-complete-refresh Recall、deadline Recall、miss/dropout、
检出延迟和处理耗时，并使用相同 catalog 大小与流式刷新协议。

## 4. 研究线 B：热词增强且保持 prefix filling

### 4.1 推荐架构

将信息分成两个明确通道：

- **稳定转写前缀通道**：只包含已经接受的 ASR transcript token，继续由现有
  commit/rollback 控制；任何热词都不能直接拼进 transcript prefix；
- **临时候选侧通道**：每次 refresh 生成一个有版本、带置信度和过期时间的候选快照，
  通过官方 `context` 构造的小动态 prompt 和/或 request-local sparse token bias，
  只影响当前可变后缀。

当前 Qwen adapter 已经通过官方 `init_streaming_state(context=...)` 取得 `prompt_raw`，
再追加稳定 `prefix_text`；这条方向是对的。主要风险不是“是否使用 context”，而是动态
context 每次变化后，是否仍能保证前缀 token 原样保留、logits processor 只处理新生成
token，以及迟到候选不会越过已提交边界。

### 4.2 先做隔离实验，再接实时检索

固定同一批候选及其到达时间，比较：

| 组别 | 动态 prompt | token bias | 用途 |
| --- | --- | --- | --- |
| B0 | 否 | 否 | 官方 prefix filling / fixed-5 基线 |
| B1 | 是 | 否 | 验证 prompt-only 的收益和前缀安全性 |
| B2 | 否 | 是 | 验证 suffix-only 稀疏偏置 |
| B3 | 是 | 是 | 验证两种注入是否互补 |

候选源按以下顺序推进：oracle 候选 → 固定检索 trace 回放 → 当前 checkpoint 实时检索 →
扩容后的 checkpoint。这样 B 线可以先研究解码机制，不被 A 线训练波动阻塞。

### 4.3 必须先验证的实现契约

- 空候选时与官方 streaming 输出逐 token 一致；
- 每次解码结果必须以传入的稳定 `prefix_text` token 开头，已提交 token 改写数为 0；
- token bias 只作用于新生成后缀，不作用于 prompt 和稳定前缀；
- token trie 能在生成后缀的任意位置启动热词路径，并能继承 prefix 末尾的部分匹配状态；
- 动态 prompt 有固定 token budget，当前可先维持最多 5 个热词；
- 迟到或低置信候选不得触发超过 `max_rollback_tokens` 的回滚；
- GLCLAP 分数先经过 dev 校准后再进入候选阈值，不能直接把相似度当概率使用；
- `RetrievalBatch -> RetrievalUpdate` 适配器记录 hotword id、rank、原始分数、校准置信度、
  观察到的 audio cutoff、chunk id 和过期策略。

当前单元测试覆盖了 token trie 的根部/连续路径和空偏置 no-op，可选 GPU 测试覆盖无热词
最终文本 parity；还需要增加真实 vLLM processor 签名、动态 context、前缀末尾部分匹配、
中途启动热词、迟到候选和多次 refresh 的集成测试。

### 4.4 热词增强的验收指标

- 主收益：BWER、目标热词 recall/precision、按实体长度和边界位置分层；
- 质量 guardrail：UWER/CER 相对无热词 baseline 退化不超过现有计划的 0.5 个百分点；
- 稳定性：已提交前缀改写数必须为 0，revision/rollback 距离受配置上界约束；
- 时效：按相同 2 秒刷新协议报告 first-complete 和 deadline 指标，同时补充更小检索刷新
  周期的消融，避免把 chunk 变小带来的时效收益误归因于模型；
- 开销：动态 prompt token 数、每次重新解码 token 数、检索/解码 P50/P95 和 RTF。

## 5. 两条线的并行与合流方式

| 研究线 | 固定输入 | 独立产出 | 合流门槛 |
| --- | --- | --- | --- |
| A：检索器扩容 | 固定 target/catalog/评测器和 2 秒刷新协议 | checkpoint、索引、逐域离线与在线检索结果 | multi-dev 改善或至少无关键域退化，补齐 global-only 与边界检验 |
| B：热词注入 | 固定 oracle/trace 候选、官方 Qwen 版本和 commit 配置 | prompt-only、bias-only、组合方案对比 | 无热词 parity、前缀改写为 0、UWER/CER guardrail 通过 |

两条线都通过各自门槛后，再把 A 的新 checkpoint 接入 B 的最佳注入方案，做一次完整端到端
矩阵。不要在端到端主实验里同时调整检索模型、置信度校准、prompt 模板和 bias 强度。

## 6. 最近一轮执行顺序

1. 冻结当前 AISHELL-NER dev/test、10k catalog、best/last checkpoint 哈希和完整评测配置；
2. 补跑 global-only 配对基线、有效的 center/cross 边界集和 real-time 回放，关闭当前缺口；
3. 扩展 manifest schema 和训练入口，实现多语种采样、同语言负例和多 dev 选择；
4. 先启动 A1 中文扩容，同时用固定 trace 开始 B0–B3 prefix-safety 实验；
5. A1 通过后加入 LibriSpeech 做 A2，并单独报告中英文结果；
6. 固定 B 线最佳注入方法后，接入实时 GLCLAP adapter；
7. 最后只做一次冻结配置的端到端 test，并保留所有 resolved config、seed、hash 和 traces。

## 7. 相关实现与设计文档

- [中文检索器扩容实验详细计划](chinese_retriever_scaling_plan.md)
- [GLCLAP 检索设计](glclap_retrieval_v1.md)
- [在线检索评测](glclap_online_evaluation.md)
- [GLCLAP 复现差异矩阵](glclap_reproduction_matrix.md)
- [流式系统设计](streaming_design.md)
- [研究规格与验收条件](spec.md)
- [整体训练评测计划](train_eval_plan.md)
- [`glclap_data.py`](../asr/contextual/glclap_data.py)
- [`train_glclap_retriever.py`](../scripts/train_glclap_retriever.py)
- [`session.py`](../asr/contextual/session.py)
- [`qwen3.py`](../asr/backends/qwen3.py)
- [`token_bias.py`](../asr/contextual/token_bias.py)
