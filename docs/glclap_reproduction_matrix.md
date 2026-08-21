# GLCLAP / AmphionASR 复现边界与项目协议

本文档防止把“借鉴方法”写成“严格复现”。当前主实验是面向 Qwen3-ASR-0.6B
累计音频检索的小闭环，不满足任一论文的完整复现条件。

## 方法对照

| 项目 | GLCLAP 原文 | AmphionASR retriever | 当前默认系统 |
|---|---|---|---|
| 声学主干 | Data2Vec2.0-large，私有中英数据预训练 | Qwen3-ASR-1.7B 合并 Stage-2 checkpoint | Qwen3-ASR-0.6B 官方 checkpoint |
| 文本主干 | multilingual BERT | Qwen LLM token embedding | Qwen LLM token embedding |
| 训练语料 | WenetSpeech、AISHELL、GigaSpeech、LibriSpeech | CommonVoice 中英 hotword 标注池 | AISHELL-1 train 小闭环 |
| global positive | 完整 transcript | 完整 transcript | 完整 transcript |
| local positive | 从 transcript 随机抽 subtext | 标注 ground-truth hotword | 每 epoch 随机 2--8 字子串 |
| local negative | 论文公式中的 batch 内其他样本 | 同语言 pool 采 4095，排除 batch positives，batch 共享 | HKUST/MagicData pool 采 4095，排除全部已说子串，batch 共享 |
| epoch | 100 + early stop | 10 | 10 |
| LR | 5e-4 | 3e-4 | 3e-4 |
| effective batch | 64 | 384 | 384 |
| precision | 未明确 | fp16 | bf16 autocast |
| temperature | 未明确 | learnable，初值 0.07 | learnable，初值 0.07 |

## 能否称为严格复现

### GLCLAP

不能。除随机 subtext 与 global-local loss 思路外，主干、训练语料、epoch、batch、
negative construction 均不同。原文还使用私有预训练音频 encoder 和私有 PhoneCall
测试集，因此仅凭论文无法做完全可验证的端到端复现。

可以报告的名称：`GLCLAP-inspired Qwen retriever` 或
`GLCLAP loss with Qwen frozen representations`。

### AmphionASR retriever

不能。我们的 4095 shared-negative 策略与其一致，但 positive 不是人工/模型挖掘的
ground-truth hotword，negative pool 也不是由同语言训练 hotword annotations 去重得到；
此外模型是 0.6B 官方 checkpoint，不是 1.7B merged Stage-2 checkpoint，精度也不同。

可以报告的名称：`Amphion-style shared-negative sampling`，不能写成
`AmphionASR reproduction`。

## `target-catalog` 为什么存在

它是评测真值注册表，解决三个问题：

1. `eval-manifest` 只保存稳定 ID，canonical/alias/metadata 集中管理；
2. stage6 需要把所有 gold targets 与 distractors 编成同一个固定大小检索库；
3. forced aligner 需要通过 target ID 找到要在 transcript/alignment items 中定位的文本。

它不应参与训练 positive 的随机抽取，也不应输入训练负词池构建。主 stage1 已拆成
training-pool 与 evaluation-catalog 两个隔离命令。

## 严格复现所需的额外实验

### GLCLAP sampling-protocol control

- 使用四个公开训练 corpus；
- local positive 按论文随机 subtext；
- local candidates 仅使用 batch 内 positive；
- 100 epochs、batch 64、LR 5e-4、early stopping；
- 如果继续使用 Qwen encoder，必须明确称为 sampling/loss protocol control，而非
  完整 GLCLAP reproduction。

### Amphion sampling-protocol control

- 对训练 split 离线挖掘 ground-truth hotwords，保留每 utterance 的明确 positive；
- 按语言从训练 annotations 构建去重 pool，不能使用 test target catalog；
- 每 batch 共享 4095 negatives，并排除 batch 所有 positives；
- 10 epochs、LR 3e-4、5% warmup、cosine、clip 1.0、batch 384、fp16；
- 使用 Qwen3-ASR-1.7B merged Stage-2 checkpoint 才能接近完整 Amphion 设置。若该
  checkpoint 不公开，应将此项列为不可复现差异。

## 论文结论的最小对照矩阵

| 实验名 | local positive | local negative | 目的 |
|---|---|---|---|
| Global-only | 无 | 无 | 检验 local objective 的增益 |
| GLCLAP protocol | random subtext | in-batch | 对齐 GLCLAP 采样方式 |
| Amphion protocol | annotated hotword | 4095 same-language shared | 对齐 Amphion 采样方式 |
| Project hybrid | random 2--8 chars | 4095 word-frequency shared | 当前低成本主系统 |

论文表格必须把这四者分开，不能只用 `GLCLAP` 一个名字覆盖不同采样协议。
