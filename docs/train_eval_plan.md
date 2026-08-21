# Training and Evaluation Plan

## Models and compute

- Qwen3-ASR-0.6B: complete development and ablation matrix
- Qwen3-ASR-1.7B: official baseline, prompt, stateful+adaptive rollback, full hybrid
- Linux CUDA single host with 4–8 GPUs of 48–80 GB; independent runs are parallelized by GPU

## Data

- Chinese primary: AISHELL-1 clean control and SpeechIO domain/long-tail set
- English replication: LibriSpeech test-clean/test-other and STOP1/STOP2
- Offline Qwen3-ForcedAligner supplies reference entity spans only for dataset preparation
- Boundary variants: center, B-400/B-200/B-100, Cross-25/50/75
- Distractors: random, homophone/near-phone, shared prefix, semantic, common word
- HKUST/MagicData word-frequency lists supply frequency-stratified common-word distractors, not gold entities
- AISHELL1-NE annotations supply target IDs, canonical forms, aliases, and split membership
- Test targets/aliases and every spoken evaluation substring are excluded from distractors
- Every catalog build archives source statistics, requested size, filters, seed, and output counts

No model weights, datasets, or generated indexes are committed to the repository.

## Probe training

1. Generate language-tagged Mandarin initial/final+tone and English ARPAbet targets.
2. Extract frozen AuT states using `extract_aut_features.py`.
3. Train only LayerNorm+Linear with CTC loss and `zero_infinity=True`.
4. Store config, symbol inventory, seed, epoch, head, and optimizer state.
5. Simulate requested chunk sizes during feature generation; use 200 ms left overlap for streaming inference.

## Matrix

- chunk: 560/1120/2000 ms; 160 ms stress only
- rollback baseline: 0/2/5/10/20; adaptive cap 32
- catalog: 10/100/1k/10k/100k
- variants: no context, prompt, trie, stateless+fixed-5,
  stateful+fixed-5, stateful+adaptive rollback, retrieval+prompt, full hybrid
- retrieval: oracle and real for every main comparison
- optional lookahead: 100/200/300 ms

## Metrics

- WER, CER, BWER, UWER
- keyword precision, recall, F1, FAR
- `BoundaryPenalty = Recall_center - Recall_cross`
- hotword stable latency, revision rate, TTFT, stable-final latency
- RTF, chunk P50/P95, GPU memory, CPU retrieval time, active states,
  prompt tokens, and re-decoded tokens per minute

Use utterance-paired bootstrap with 2000 resamples and seed 42. The main claim
passes only when cross-boundary BWER and BoundaryPenalty improve over
stateless+fixed-5 with a 95% confidence interval excluding zero, while absolute
UWER degradation is at most 0.5 percentage points.

## Reproduction

```bash
build_glclap_catalogs \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --negative-output data/hotwords/zh_train_10k.jsonl \
  --target-catalog data/aishell1_ne/targets_test.jsonl \
  --eval-manifest data/aishell1_ne/test.jsonl \
  --evaluation-output data/hotwords/aishell1_ne_10k.jsonl \
  --size 10000 --seed 42

python -m unittest discover -s tests -v
RUN_CONTEXTUAL_STRESS=1 python -m unittest tests.test_contextual_stress -v
python scripts/decode_streaming_contextual.py --config configs/contextual/main.yaml --catalog data/hotwords.jsonl --manifest data/test.jsonl --output outputs/main.jsonl --trace-dir outputs/traces
python scripts/eval_contextual.py --input outputs/main.jsonl --baseline outputs/baseline.jsonl --output outputs/metrics.json
```

Real Qwen integration is gated by `QWEN_MODEL_PATH` and the pinned optional
dependencies; dependency-free CI does not download weights.
