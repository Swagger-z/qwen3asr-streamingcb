# GLCLAP 在线及时检出与延迟评测

只评测已有检索器，不训练、不接入 prompt、ASR decoder 或 commit/rollback。
检出定义为进入 Top-K，报告 K=1/5/10/20/50，不增加分数阈值。
原有句末指标保留，新增指标放在汇总 JSON 的 online 字段。

## 快速开始

在服务器仓库根目录执行。前提是已有 AISHELL-NER gold 转换产物和离线对齐文件：

~~~bash
CUDA_VISIBLE_DEVICES=4 SPLIT=test \
EXP_DIR=/data/zhengjie/research/qwen3asr-posttraining/qwen3asr-streamingcb/outputs/glclap/frozen_bs8_100epoch \
bash run_online_eval.sh all
~~~

默认 best.pt、快速回放、原句和七种边界样本、2秒 refresh、100ms feed、
Top-50、预热3次。SPLIT=dev 切换为 dev 的独立输入输出。
CHECKPOINTS="best last" 运行两个 checkpoint；
REPLAY_MODES="fast realtime" 运行两种回放。组合按顺序执行，每进程一张 GPU。

如果没有对齐文件，加 RUN_ALIGNER=1：仅在 ALIGNED_MANIFEST 不存在时调用已有
离线 aligner。ALIGNER_MODEL 可以指定本地权重。不会重新生成 NE 标签或训练模型。

~~~bash
# 无数据、无模型、无文件写入的命令检查
DRY_RUN=1 SPLIT=dev CHECKPOINTS="best last" REPLAY_MODES="fast realtime" \
bash run_online_eval.sh all
~~~

## 1. 外部输入的格式和来源

默认 DATA_ROOT=/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets，
AISHELL_NER_DIR=DATA_ROOT/AISHELL-NER，所有路径均可用环境变量覆盖。

| 变量 | 默认位置 | 格式、来源 |
|---|---|---|
| SOURCE_MANIFEST | AISHELL_NER_DIR/<split>_entities.jsonl | gold 转换产物，每个含实体原句一行，包含全部 mention |
| FULL_MANIFEST | AISHELL_NER_DIR/<split>.jsonl | 同次转换的全量原句，包括无实体句，仅新建 distractor 库时需要 |
| TARGET_CATALOG | AISHELL_NER_DIR/targets_<split>.jsonl | gold 实体注册表，ID 与清单一致，不是训练正样本表 |
| ALIGNED_MANIFEST | AISHELL_NER_DIR/<split>_aligned.jsonl | 独立 aligner 输出，每 mention 一行，带 hotword_start/end_sec |
| HKUST_WORD_FREQ | DATA_ROOT/hkust_wo_st/word_freq.txt | 外部 UTF-8，每行“词语 正整数频次”，用于 distractor |
| MAGICDATA_WORD_FREQ | DATA_ROOT/magicdata/word_freq.txt | 同上 |
| EXP_DIR | outputs/glclap/frozen_bs8_100epoch | 已有训练目录，包含 best.pt/last.pt |
| CONFIG | configs/glclap/qwen_post_projector_frozen.yaml | 与 checkpoint 对应的配置；消融需指定其配置 |
| QWEN_MODEL | /data/zhengjie/resources/pretrain_models/Qwen3-ASR-0.6B | 已有冻结 Qwen 权重 |
| CATALOG | RUN_DIR/data/catalog_10000.jsonl | 可复用已有评测库，否则 stage2 构建 |
| INDEX_PATH | 不指定 | 可复用 .npz 及其 .npz.json；只允许单个 checkpoint |

SOURCE_MANIFEST 示例（一行一个 JSON，同词重复 mention 不去重）：

~~~json
{"key":"u1","source":"/data/u1.wav","target":"张三见到张三","target_hotword_ids":["person-1"],"entities":[{"mention_id":"u1#0","hotword_id":"person-1","text":"张三","char_start":0,"char_end":2,"occurrence_index":0},{"mention_id":"u1#1","hotword_id":"person-1","text":"张三","char_start":4,"char_end":6,"occurrence_index":1}]}
~~~

TARGET_CATALOG/CATALOG 示例：

~~~json
{"catalog_version":"aishell-ner-test-targets-v1","id":"person-1","text":"张三","aliases":[],"language":"zh","weight":1.0,"metadata":{"split":"test"}}
~~~

ALIGNED_MANIFEST 的单个 mention 映射示例（完整文件还须有 u1#1）：

~~~json
{"key":"u1__mention_unique","utt_id":"u1__mention_unique","source_utt_id":"u1","source":"/data/u1.wav","audio":"/data/u1.wav","target":"张三见到张三","target_hotword_ids":["person-1"],"aligned_hotword_id":"person-1","aligned_mention_id":"u1#0","hotword_start_sec":1.8,"hotword_end_sec":2.2}
~~~

对齐文件必须覆盖 SOURCE_MANIFEST 的每个 mention。stage0 按
source_utt_id+aligned_mention_id 升级旧 aligner 重复 key，不猜测缺失实体。
缺少 gold 转换产物时使用 README 的 prepare_aishell_ner.py 命令，dev/test 分开；
这些标签直接来自 AISHELL-NER，不由词表匹配或 aligner 推断。

## 2. 每个 stage 的输入输出

RUN_DIR 默认 EXP_DIR/online_eval_<split>。本入口的编号独立于训练 run.sh。

| Stage | 输入 | 输出 | 依赖 |
|---|---|---|---|
| stage0 | SOURCE_MANIFEST、ALIGNED_MANIFEST、TARGET_CATALOG | data/original_timed.jsonl、focus_timed.jsonl、preparation_report.json | 默认 CPU；缺对齐且 RUN_ALIGNER=1 才用 GPU aligner |
| stage1 | stage0 的 focus_timed.jsonl | data/boundary.jsonl、boundary_wav/ | CPU；DATASETS=original 跳过 |
| stage2 | 已有 CATALOG，或两个词频表+TARGET_CATALOG+FULL_MANIFEST；checkpoint/config/Qwen | catalog 和报告（若需构建）、<tag>/index.npz 及 .npz.json | 建库 CPU、建索引 GPU；不训练 |
| stage3 | stage0 原句/stage1 边界清单、index、对应 checkpoint/config | <tag>/<dataset>_<replay>.jsonl、.jsonl.run.json、*_traces/ | GPU；校验时标、目标覆盖、WAV 和索引哈希 |
| stage4 | stage3 检索结果 | *_metrics.json、*_entities.jsonl、*_refreshes.jsonl | CPU；默认2000次按原句分组的 bootstrap |

original_timed 每原句一行，保留全部实体时标，音频只执行一轮回放。
focus_timed 每 mention 一行，也保留全部时标，但 target_hotword_ids 仅包含焦点。
它只用于生成边界变体，不用来重复检索原句。

工具生成的 schema v2 包含：

- key/utt_id：一致且唯一；变体由原句、mention、边界条件确定。
- source/audio：一致，指向实际输入 WAV；变体必须指向加静音后的 WAV。
- source_utt_id、focus_mention_id、boundary_group：显式身份，不从文件名猜测。
- entities[].mention_id/hotword_id/start_sec/end_sec：相对当前 WAV 的秒数。
- timing_schema_version=2、duration_sec、audio_sha256。
- 边界额外保留 original_audio/original_audio_sha256/original_duration_sec、
  leading_silence_sec、boundary_chunk_sec。
- hotword_start/end_sec 与 word_start/end_sec 若存在，都指变体焦点的相同时标。

所有实体随静音整体平移。仅接受16kHz PCM16 WAV。
范围、唯一ID、实体覆盖、路径、哈希或边界条件不合法均报错，不输出零延迟。

七种条件延续既有实验口径：Center 的中点在 chunk 中央且整个实体位于 chunk 内；
B-400/200/100 指实体中点在边界前对应毫秒数，不保证长实体末尾也在边界前；
Cross-25/50/75 指实体时长的对应比例在边界之后。
静音按 WAV 采样点量化。生成 chunk 必须与检索 chunk 相同。
实体过长无法构造合法 Center 时明确失败，应增大 chunk 或事先声明研究子集，
不能根据检索命中情况筛选。

## 3. 回放与延迟分解

100ms feed 是 PCM 交付粒度，不是检索频率。2s refresh 表示在2、4、6秒……
重新编码全部累计音频；真实音频尾块立即处理，整 chunk 结束不重复检索。

| 字段 | 含义 |
|---|---|
| audio_cutoff_sec = t_j | 本次覆盖的累计音频终点 |
| ready_sec = a_j | 按 feed 计划凑齐输入的时刻；尾块不等下个完整 feed |
| start_sec = s_j | 开始处理 |
| finish_sec = f_j | GPU 工作完成且 CPU 命中列表可用 |
| processing_sec | 完整检索服务时间，包含 AuT/projector/adapter/search |
| feed_wait_sec、queue_wait_sec | a_j-t_j、s_j-a_j |

快速回放不等待真实时间，以测得的每次服务时间推演单流 FIFO：
s_j=max(a_j,f_(j-1))，f_j=s_j+processing_sec。
clock=simulated_fifo，结果可用时刻是模拟值，不能宣称为真实墙钟延迟。

真实定速回放按单调时钟的绝对交付时刻等待，clock=monotonic_realtime。
计算落后则保留积压 refresh，按顺序处理，不“计算后再等100ms”。
音频已预加载到内存，这不是声卡/网络流实验。每句重新设定时间原点。

实体结束时间 e 后，第一个 t_j>=e 且目标进入 Top-K 的 refresh 定义有效检出：

~~~text
总延迟 = f_j-e
       = (t_j-e) + (a_j-t_j) + (s_j-a_j) + (f_j-s_j)
       = 音频积累等待 + feed等待 + 排队/调度等待 + 检索计算
~~~

实体1.8–2.2秒、4秒首次有效命中、计算0.1秒且无排队时：
积累等待1.8秒，总延迟1.9秒，不是0.1秒。

模型/索引加载、WAV读取、哈希与校验、预热、结果写盘和额外 offline 验证
不计入发现延迟。默认用第一句前一个 chunk 预热3次，不计正式刷新次数。
离线 aligner 时间戳只供评测，在线检索器拿不到实体真值。

## 4. 指标口径与分母

每原句每个 hotword_id 只用首次出现计算发现指标；重复 mention 只报候选可用性。
边界数据只评测 focus，但用完整实体序列判断它是否首次出现。

| online.by_k.<K> 字段 | 定义 |
|---|---|
| target_count/detected_count/missed_count | 首次实体数、最终前曾有效检出的数、从未有效检出的数 |
| first_complete_refresh_recall | 首个覆盖完整发音的 refresh 是否命中；分母全部首次目标 |
| deadline_recall.<ms> | f_j<=e+deadline 的有效命中率；漏检和超时都在分母 |
| miss_rate | 从未在完整发音后命中，不等于最终榜单不含目标 |
| latency_ms_mean/p50/p95 | 首次有效检出的 f_j-e；仅成功样本，数量为 detected_count |
| audio_wait/feed_wait/queue_wait/processing_ms_* | 对同一成功集合做四项分解 |
| early_before_onset_rate/early_during_word_rate | 发音前/发音过程中提前进入Top-K，两个集合可以重叠 |
| dropout_rate | 检出后有后续 refresh 可观察者中，曾掉出的比例；分母 post_detection_observed_count |
| repeat_candidate_availability | 重复 mention 结束后的首个 refresh 已有候选的比例，分母 repeat_mention_count |

默认 deadline 为0/100/200/500/1000/2000ms，可用 DEADLINES_MS 覆盖。
未检出 latency_ms=null；全漏检时均值和分位数也是 null，不能填0。
提前候选不能产生负发现延迟，必须在完整发音后的 refresh 仍命中才算有效。
重复实体没有发现延迟，不把历史候选误当再次识别。

online.by_boundary.<K> 按七种条件分别报告上述及时召回和延迟。
online.boundary_penalty.<K> 为 Center 减三种 Cross 平均，分别给
first_complete_refresh 和各 deadline 的召回差及95% CI。数值乘100为百分点。
只使用 Center+三种 Cross 齐全的同原句/同 mention 配对；
unpaired_focus_count 显式报告缺失，不能以零替代。
bootstrap 按 source_utt_id 抽样，保留该原句的全部实体和变体，默认2000次、seed42，
不将七种变体作为独立语音。原顶层 boundary_penalty 仍是句末口径。

*_entities.jsonl 每 mention × K 一行，包含起止时标、primary、首完整 refresh、
首有效检出 chunk、提前/漏检/掉出标志、deadline结果及延迟分解。
*_refreshes.jsonl 每 refresh 一行，包含 t/a/s/f、分阶段 timings_ms、
frame_count，及以此时已完整发音的首次实体为分母的 Recall@K；暂无完整目标时为 null。

online.compute 汇总所有 refresh 的服务耗时、P50/P95、processing_rtf、
refresh数和累计被编码音频秒数（sum t_j）。
这里没有计算相对 offline/完整 ASR 的额外开销，不能作为增量计算贡献。
顶层 rtf_mean 沿用回放耗时/音频时长；定速回放包含等待，不能作纯计算RTF。
原有顶层阶段耗时P50/P95仍是句末口径，全refresh分段耗时见明细。

## 5. 复用文件与重跑范围

复用已有 best index 时，CATALOG 必须是建该索引时的同一份文件。
不同 checkpoint 不混用 index，程序核对 .npz.json 中的 checkpoint/catalog SHA256：

~~~bash
CUDA_VISIBLE_DEVICES=4 SPLIT=test CHECKPOINTS=best \
EXP_DIR=/data/experiment/frozen_bs8_100epoch \
CATALOG=/data/hotwords/aishell_ner_10k.jsonl \
INDEX_PATH=/data/experiment/frozen_bs8_100epoch/aishell_ner_10000_bestpt.npz \
RUN_DIR=/data/experiment/frozen_bs8_100epoch/online_test_v2 \
REPLAY_MODES="fast realtime" \
bash run_online_eval.sh all
~~~

默认保护已有产物，推荐新 RUN_DIR。OVERWRITE=1 显式覆盖所选stage产物，
不会删除旧目录；不再生成的旧文件可能残留，正式实验宜使用新目录。

| 改动 | 重跑范围 |
|---|---|
| deadline/bootstrap配置 | stage4，不重跑GPU |
| replay模式/feed粒度/预热 | stage3、stage4 |
| refresh chunk | stage1重建边界、stage3、stage4；原句时标、索引可复用 |
| checkpoint/adapter配置 | stage2建对应index、stage3、stage4 |
| catalog内容 | stage2重建index、stage3、stage4 |
| 标注/alignment/WAV | 必要的重对齐，再stage0、stage1、stage3、stage4；实体ID变化还需stage2 |

~~~bash
DEADLINES_MS="0 100 200 500 1000 2000 3000" OVERWRITE=1 \
EXP_DIR=/data/experiment/frozen_bs8_100epoch SPLIT=test \
bash run_online_eval.sh stage4
~~~

旧结果不加 --online 仍能计算句末指标。
旧边界 source/audio 错误导致实际读取原WAV的结果，必须重生成并重新检索，
不能贴新时标复用旧排名。
仅 key、音频SHA256、路径、时长、目标和身份都可核验的结果，才允许通过
eval_hotword_retrieval.py --online --timing-manifest 补充标注；
仍需完整逐refresh服务时钟和配置。无计时的旧trace无法恢复真实发现延迟。

## 6. 复现快照和验证

.jsonl.run.json 保存配置、依赖版本、checkpoint/index/catalog/manifest哈希、
实现文件哈希、设备、预热和排除计时范围。
结果逐句保留实际WAV哈希，preparation_report.json保留时标来源哈希；
metrics保留输入结果/运行快照哈希及评测实现哈希。

CPU回归：

~~~bash
python -m unittest discover -s tests -p 'test_online*.py' -v
RUN_CONTEXTUAL_STRESS=1 RUN_GLCLAP_STRESS=1 python -m unittest discover -s tests -v
~~~

服务器真实GPU集成检查（16kHz WAV至少1秒，fixture最多取2.1秒）：

~~~bash
CUDA_VISIBLE_DEVICES=4 RUN_GLCLAP_GPU_REPLAY=1 \
QWEN_MODEL_PATH=/data/zhengjie/resources/pretrain_models/Qwen3-ASR-0.6B \
QWEN_TEST_WAV=/data/example.wav \
GLCLAP_CHECKPOINT=/data/experiment/best.pt \
GLCLAP_INDEX=/data/experiment/index.npz \
python -m unittest discover -s tests -p test_online_runner.py -v
~~~

该检查验证两种回放逐refresh输入及Top-K ID排序一致、时间分解成立。
本地可控时钟和CPU测试只验证逻辑，不能代替服务器实测延迟/吞吐结论。
