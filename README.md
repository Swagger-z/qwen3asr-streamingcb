# Qwen3-ASR Stateful Contextual Biasing

Research implementation of cross-chunk contextual biasing for Qwen3-ASR-style
streaming LLM-ASR. The central mechanism keeps unresolved hotword paths alive
across chunks and uses them to control the stable prefix and bounded rollback.

The original CTC skeleton remains available under its existing modules. New
LLM-ASR work lives in `asr.backends` and `asr.contextual`.

## Implemented

- Versioned JSONL hotword catalogs and per-session ID filters/overlays
- Memory-conscious token/phoneme trie with Aho-Corasick failure links
- Stateful pruned CTC posterior token passing with blank/repeat handling
- Candidate `active/confirmed/rejected/expired` lifecycle with hysteresis and TTL
- Official fixed-5-compatible and hotword-aware bounded rollback/commit controller
- Multi-path tokenizer trie and confidence-gated sparse vLLM logits processor
- Accumulated-audio Qwen3-ASR adapter with strict dependency version checks
- GLCLAP dual-adapter retrieval, Qwen-projector initialization ablations, exact 10k Top-50 index, and streaming retrieval CLI
- Deterministic HKUST/MagicData-style word-frequency merging and leakage-safe 10k catalog preparation
- Transcript probe, generic frozen-AuT sidecar, phoneme head training/extraction tools
- Streaming session, JSONL traces, boundary-stress WAV builder, BWER/UWER and bootstrap evaluation

## Environment

The algorithmic core runs on Python 3.10+ without PyTorch. Official Qwen
streaming requires Linux CUDA and the pinned runtime:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-qwen.txt
```

The reproducible integration target is `qwen-asr==0.0.6`,
`transformers==4.57.6`, and `vllm==0.14.0`. Upgrades must pass the adapter
contract test before changing these pins.

## Hotword catalog

One JSON object per line:

```json
{"catalog_version":"paper-v1","id":"poi-1","text":"张江人工智能岛","aliases":["人工智能岛"],"language":"zh","weight":1.0,"metadata":{"pronunciations":[["zh:i:zh","zh:f:ang1","zh:i:j","zh:f:iang1"]]}}
```

For M2 runs, build pronunciation paths explicitly with pinyin/ARPAbet. Character
fallback exists only for M1 and dependency-free tests.

## Prepare GLCLAP catalogs

Corpus `word_freq.txt` files contain `TERM COUNT` rows. They are distractor
sources, not gold named-entity annotations. Merge them into a training-negative
catalog and combine annotated AISHELL1-NE targets with clean distractors:

```bash
build_glclap_catalogs \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --negative-output data/hotwords/zh_train_10k.jsonl \
  --target-catalog data/aishell1_ne/targets_test.jsonl \
  --eval-manifest data/aishell1_ne/test.jsonl \
  --evaluation-output data/hotwords/aishell1_ne_10k.jsonl \
  --report data/hotwords/catalog_build_report.json \
  --size 10000 \
  --seed 42
```

The builder keeps 2--8 Han-character terms, samples across frequency-rank
buckets, excludes targets/aliases and spoken evaluation substrings, validates
manifest target IDs, and records source counts in both catalog metadata and a
JSON report. Training also excludes every spoken 2--8 character batch substring
before sampling shared negatives. See `docs/glclap_catalog_preparation.md` for
schemas, leakage rules, and the boundary-manifest workflow.

## Decode and evaluate

The manifest schema is JSONL with `utt_id`, `audio`, optional `text`,
`hotword_ids`, `context`, and `boundary_group`.

```bash
python scripts/decode_streaming_contextual.py \
  --config configs/contextual/main.yaml \
  --catalog data/hotwords.jsonl \
  --manifest data/test.jsonl \
  --output outputs/contextual/hypotheses.jsonl \
  --trace-dir outputs/contextual/traces

python scripts/eval_contextual.py \
  --input outputs/contextual/hypotheses.jsonl \
  --baseline outputs/official/hypotheses.jsonl \
  --output outputs/contextual/metrics.json
```

Use `configs/qwen/official_streaming.yaml` for the no-context fixed-5 baseline.

## Phoneme probe

First extract frozen AuT states from a manifest containing `target_ids`, then
train only the LayerNorm+Linear CTC head:

```bash
python scripts/extract_aut_features.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --manifest data/probe_train.jsonl \
  --output-dir outputs/probe/features \
  --output-manifest outputs/probe/features.jsonl

python scripts/train_phoneme_probe.py \
  --config configs/probe/phoneme_ctc.yaml \
  --manifest outputs/probe/features.jsonl \
  --output outputs/probe/phoneme_head.pt
```

## Boundary stress set

The input manifest must contain offline reference fields
`hotword_start_sec`/`hotword_end_sec`:

```bash
python scripts/build_boundary_stress.py \
  --manifest data/aligned_hotwords.jsonl \
  --output-dir data/boundary_wav \
  --output-manifest data/boundary_test.jsonl \
  --chunk-ms 2000
```

This creates Center, B-400/B-200/B-100, and Cross-25/50/75 variants by
prepending silence; the online runtime never invokes forced alignment.

## Tests

```bash
python -m compileall asr scripts tests
python -m unittest discover -s tests -v
RUN_CONTEXTUAL_STRESS=1 python -m unittest tests.test_contextual_stress -v
QWEN_MODEL_PATH=/models/Qwen3-ASR-0.6B QWEN_TEST_WAV=data/smoke.wav \
  python -m unittest tests.test_qwen_contract tests.test_qwen_output_parity -v
```

See `docs/spec.md`, `docs/streaming_design.md`, and
`docs/train_eval_plan.md` for interfaces, state transitions, and the paper
evaluation matrix.
