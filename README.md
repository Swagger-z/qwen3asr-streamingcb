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
- Single-node multi-GPU GLCLAP training with torchrun/DDP, global-batch preservation, rank-sharded train/dev data, all-rank validation, packed audio encoding, and frozen-feature caches
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
sources, not gold named-entity annotations. Evaluation entities come directly
from the official AISHELL-NER PER/LOC/ORG markers; no lexicon matching infers
the labels.

First convert the gold annotation into the project's catalog and manifest views:

```bash
python scripts/prepare_aishell_ner.py \
  --annotated-transcript /data/AISHELL-NER/data/aishell_ner_transcript.test.txt \
  --wav-root /data/AISHELL-1/wav/test \
  --target-catalog-output data/aishell_ner/targets_test.jsonl \
  --eval-manifest-output data/aishell_ner/test.jsonl \
  --entity-manifest-output data/aishell_ner/test_entities.jsonl \
  --report data/aishell_ner/test_preparation_report.json \
  --split test
```

Training-pool and evaluation-catalog construction remain intentionally separate:

```bash
python scripts/build_glclap_training_pool.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --output data/hotwords/zh_train_10k.jsonl \
  --report data/hotwords/training_pool_report.json \
  --size 10000 --seed 42

python scripts/build_glclap_evaluation_catalog.py \
  --word-freq hkust=/data/hkust/word_freq.txt \
  --word-freq magicdata=/data/magicdata/word_freq.txt \
  --target-catalog data/aishell_ner/targets_test.jsonl \
  --eval-manifest data/aishell_ner/test.jsonl \
  --output data/hotwords/aishell_ner_10k.jsonl \
  --report data/hotwords/evaluation_catalog_report.json \
  --size 10000 --seed 42
```

The training command never reads evaluation annotations. The evaluation command
keeps every annotated target, excludes spoken evaluation substrings from its
distractors, and validates target IDs. Training additionally excludes every
spoken 2--8 character batch substring before sampling shared negatives. See
`docs/aishell_ner_preparation.md` for the gold-label conversion,
`docs/glclap_stage_data_contracts.md` for every stage's file contracts, and
`docs/glclap_reproduction_matrix.md` for the exact GLCLAP/Amphion differences.

## Decode and evaluate

Dataset-facing manifests use one JSON object per line with the canonical fields
`key`, `source`, and `target`, for example:

```json
{"key":"BAC009S0002W0122","source":"/data/aishell/wav/BAC009S0002W0122.wav","target":"而对楼市成交抑制作用最大的限购"}
```

The readers also accept the legacy aliases `utt_id`, `audio`, and `text` for
already-generated internal artifacts. Optional contextual fields include
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

## Offline hotword alignment

`AISHELL_NER_ENTITY_MANIFEST` already contains gold entity text, type, character
span, and mention ID parsed from AISHELL-NER. The independent forced aligner only
adds acoustic timestamps needed by the boundary experiment:

```bash
export AISHELL_NER_ENTITY_MANIFEST=/data/aishell_ner/test_entities.jsonl
export AISHELL_NER_TARGET_CATALOG=/data/aishell_ner/targets_test.jsonl
export AISHELL_NER_ALIGNED_MANIFEST=/data/aishell_ner/test_aligned.jsonl
export CUDA_VISIBLE_DEVICES=0

INSTALL_DEPS=1 bash run_aligner.sh all
```

`run_aligner.sh` validates the pinned runtime, aligns the marker-free transcript,
matches each gold mention, and verifies that timestamps fit the PCM16 WAV. Every
entity mention becomes one aligned record. Repeated occurrences of the same
surface form remain distinct and are resolved by their annotated
`occurrence_index`. The aligner never creates entity labels and is never loaded
by streaming retrieval.

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

## End-to-end staged experiment

`run.sh` connects gold-label conversion, catalog preparation, boundary-set
construction, GLCLAP training, indexing, accumulated-audio retrieval, evaluation,
and tests:

```bash
export HKUST_WORD_FREQ=/data/hkust/word_freq.txt
export MAGICDATA_WORD_FREQ=/data/magicdata/word_freq.txt
export AISHELL1_TRAIN_MANIFEST=/data/aishell1/train.jsonl
export AISHELL_NER_DEV_ANNOTATED_TRANSCRIPT=/data/AISHELL-NER/data/aishell_ner_transcript.dev.txt
export AISHELL_NER_DEV_WAV_ROOT=/data/AISHELL-1/wav/dev
export AISHELL_NER_ANNOTATED_TRANSCRIPT=/data/AISHELL-NER/data/aishell_ner_transcript.test.txt
export AISHELL_NER_WAV_ROOT=/data/AISHELL-1/wav/test

# Build train negatives and parse AISHELL-NER dev/test gold entities separately.
bash run.sh stage0 stage1
# Add offline acoustic timestamps for the boundary experiment.
INSTALL_DEPS=1 bash run_aligner.sh all
# Continue with boundary generation, training, indexing, retrieval, and evaluation.
bash run.sh stage2 stage3 stage4 stage5 stage6 stage7 stage8 stage9
```
```bash
# A single experiment on four visible GPUs; global batch remains 384.
# Put frozen Qwen features on fast shared storage so later epochs skip AuT.
FEATURE_CACHE_DIR=/fast/glclap_qwen_cache \
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 RUN_ABLATIONS=0 \
  bash run.sh stage4
```


See `docs/glclap_runbook.md` for every stage, input schema, resume behavior,
ablation switch, and output path. Validation uses non-overlapping AISHELL-NER dev
gold entities and can run at epoch end or every fixed number of optimizer updates;
downstream indexing uses the best Recall@50 checkpoint.
Training defaults to packed variable-length Qwen audio encoding, vectorized parallel
WAV loading, cached frozen text embeddings, and a persistent pre/post-projector
feature cache. Set `AUDIO_BATCHING=serial` for the official per-audio precision path.

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
