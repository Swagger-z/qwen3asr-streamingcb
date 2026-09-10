# Multilingual GLCLAP Retriever Runbook

This pipeline expands retriever training to AISHELL-1, AISHELL-2, MagicData,
HKUST, and LibriSpeech 960 while keeping the Chinese-only `run.sh` unchanged.
The entrypoint is `run_multilingual_glclap.sh`.

## Data contract

Every normalized training row has `key`, `source_key`, `source`, `target`,
`language`, `corpus`, and `split`. `language` is `zh` or `en`; `key` is exactly
`<corpus>:<source_key>`. Evaluation rows add `entities[].text`,
`entities[].hotword_id`, and `target_hotword_ids`.

Training JSONL is byte-offset indexed. A full epoch visits each record once. A
positive `training.samples_per_epoch` uses deterministic largest-remainder
quotas over each `corpus/split`, followed by a deterministic global shuffle and
DDP sharding. The checkpoint records training-manifest, negative-pool and
validation hashes and rejects incompatible resume attempts.

Chinese local positives contain 2--8 contiguous characters. English positives
contain 1--4 complete words and matching is NFKC/case-insensitive with strict
word boundaries. A mixed batch uses one cross-language global loss and separate
same-language local losses, each with 4,095 language-specific negatives.

## Stages

```bash
bash run_multilingual_glclap.sh stage0 stage1 stage2 stage3 stage4
bash run_multilingual_glclap.sh stage5
bash run_multilingual_glclap.sh stage6 stage7 stage8
```

- `stage0`: verify the branch and all required inputs.
- `stage1`: convert AISHELL-2 and create the normalized training mix.
- `stage2`: build independent Chinese and English train-only negative pools.
- `stage3`: freeze LibriSpeech synthetic rare-phrase labels and optional STOP labels.
- `stage4`: build Chinese 10k, English 10k, and bilingual 20k catalogs.
- `stage5`: train paired global-only and frozen-projector variants.
- `stage6`: build all model/catalog indexes.
- `stage7`: run offline and 2-second streaming evaluation against both mono and
  bilingual indexes, then write `multilingual-summary.json`.
- `stage8`: run the dependency-free unit-test suite.

By default `COMPUTE_MATCHED=1` sets each logical epoch to the AISHELL-1 record
count. Set `COMPUTE_MATCHED=0 SAMPLES_PER_EPOCH=0` for ten complete passes over
the mixed manifest. `NUM_GPUS`, `GLOBAL_BATCH_SIZE`, every input path and every
output root are environment-overridable.

## Validation and evaluation

`best.pt` maximizes `macro_language_recall_at_50`: AISHELL-NER supplies the
Chinese value, the average of LibriSpeech dev-clean/dev-other supplies the
English value, and the two languages receive equal weight. STOP1/STOP2 are test
only.

Each test set is evaluated with its 10k monolingual index and with the bilingual
20k index. Reports include the paired mixed-catalog Recall@50 penalty and, for
the frozen model, paired gain over the global-only model. When an audited timing
manifest is provided through the corresponding `*_TIMING_MANIFEST` variable,
streaming reports also include first-complete-refresh, deadline, dropout and
latency metrics.

STOP conversion accepts the official `wav.scp`, `hotlists`, and
`hotlists.uniq`. Without a companion transcript it deliberately emits
retrieval-only rows; timed streaming evaluation remains disabled until a
transcript and forced alignment are available.

LibriSpeech targets are constrained to a deterministic 8k subset of the
train-derived English term pool, leaving catalog capacity for all STOP targets
while keeping the final English catalog at exactly 10k entries.
