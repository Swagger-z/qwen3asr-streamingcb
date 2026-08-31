#!/usr/bin/env bash
# Standalone online retrieval evaluation. No training or decoder integration.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets}"
AISHELL_NER_DIR="${AISHELL_NER_DIR:-${DATA_ROOT}/AISHELL-NER}"
SPLIT="${SPLIT:-test}"
EXP_DIR="${EXP_DIR:-${REPO_ROOT}/outputs/glclap/frozen_bs8_100epoch}"
RUN_DIR="${RUN_DIR:-${EXP_DIR}/online_eval_${SPLIT}}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${AISHELL_NER_DIR}/${SPLIT}_entities.jsonl}"
FULL_MANIFEST="${FULL_MANIFEST:-${AISHELL_NER_DIR}/${SPLIT}.jsonl}"
TARGET_CATALOG="${TARGET_CATALOG:-${AISHELL_NER_DIR}/targets_${SPLIT}.jsonl}"
ALIGNED_MANIFEST="${ALIGNED_MANIFEST:-${AISHELL_NER_DIR}/${SPLIT}_aligned.jsonl}"
TIMED_MANIFEST="${TIMED_MANIFEST:-${RUN_DIR}/data/original_timed.jsonl}"
FOCUS_MANIFEST="${FOCUS_MANIFEST:-${RUN_DIR}/data/focus_timed.jsonl}"
BOUNDARY_MANIFEST="${BOUNDARY_MANIFEST:-${RUN_DIR}/data/boundary.jsonl}"
BOUNDARY_WAV_DIR="${BOUNDARY_WAV_DIR:-${RUN_DIR}/data/boundary_wav}"
CATALOG_SIZE="${CATALOG_SIZE:-10000}"
CATALOG="${CATALOG:-${RUN_DIR}/data/catalog_${CATALOG_SIZE}.jsonl}"
HKUST_WORD_FREQ="${HKUST_WORD_FREQ:-${DATA_ROOT}/hkust_wo_st/word_freq.txt}"
MAGICDATA_WORD_FREQ="${MAGICDATA_WORD_FREQ:-${DATA_ROOT}/magicdata/word_freq.txt}"
CONFIG="${CONFIG:-configs/glclap/qwen_post_projector_frozen.yaml}"
QWEN_MODEL="${QWEN_MODEL:-/data/zhengjie/resources/pretrain_models/Qwen3-ASR-0.6B}"
ALIGNER_MODEL="${ALIGNER_MODEL:-Qwen/Qwen3-ForcedAligner-0.6B}"
RUN_ALIGNER="${RUN_ALIGNER:-0}"
CHECKPOINTS="${CHECKPOINTS:-best}"
REPLAY_MODES="${REPLAY_MODES:-fast}"
DATASETS="${DATASETS:-original boundary}"
INDEX_PATH="${INDEX_PATH:-}"
CHUNK_MS="${CHUNK_MS:-2000}"
FEED_STEP_MS="${FEED_STEP_MS:-100}"
TOP_K="${TOP_K:-50}"
WARMUP_REFRESHES="${WARMUP_REFRESHES:-3}"
DEADLINES_MS="${DEADLINES_MS:-0 100 200 500 1000 2000}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-42}"
VERIFY_OFFLINE="${VERIFY_OFFLINE:-0}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'HELP'
Usage: bash run_online_eval.sh [all | stage0 stage1 stage2 stage3 stage4]
 stage0  Validate/upgrade alignments to grouped and focused timed manifests.
 stage1  Create audited seven-condition boundary WAVs (when DATASETS includes boundary).
 stage2  Build/reuse a fixed catalog and checkpoint-specific indexes.
 stage3  Retrieve each dataset with fast and/or realtime replay.
 stage4  Evaluate online discovery, latency, boundary pairs and detailed records.

Defaults: SPLIT=test CHECKPOINTS=best DATASETS="original boundary" REPLAY_MODES=fast.
Set REPLAY_MODES="fast realtime" for both. One process uses one visible GPU.
SOURCE_MANIFEST, FULL_MANIFEST and TARGET_CATALOG come from AISHELL-NER preparation.
ALIGNED_MANIFEST comes from the offline aligner. RUN_ALIGNER=1 generates it if missing.
EXP_DIR contains best.pt/last.pt. INDEX_PATH may reuse one index with one checkpoint.
Existing outputs are protected; use a new RUN_DIR or explicitly set OVERWRITE=1.
DRY_RUN=1 prints all commands without requiring input files or writing outputs.
See docs/glclap_online_evaluation.md for formats, defaults and interpretation.
HELP
}
run_cmd() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}
require_file() {
  if [[ "$DRY_RUN" != 1 && ! -f "$1" ]]; then
    echo "[missing] $1; run the prerequisite stage or override the path" >&2
    exit 2
  fi
}
protect_output() {
  if [[ "$DRY_RUN" != 1 && "$OVERWRITE" != 1 && -e "$1" ]]; then
    echo "[exists] $1; choose a new RUN_DIR or set OVERWRITE=1" >&2
    exit 2
  fi
}
dataset_manifest() {
  case "$1" in original) printf '%s' "$TIMED_MANIFEST";; boundary) printf '%s' "$BOUNDARY_MANIFEST";; esac
}
index_for() {
  if [[ -n "$INDEX_PATH" ]]; then printf '%s' "$INDEX_PATH"
  else printf '%s/%s/index.npz' "$RUN_DIR" "$1"; fi
}
stage0() {
  echo '[stage0] prepare audited original/focus entity timestamps'
  require_file "$SOURCE_MANIFEST"
  require_file "$TARGET_CATALOG"
  if [[ ! -f "$ALIGNED_MANIFEST" && "$RUN_ALIGNER" == 1 ]]; then
    run_cmd "$PYTHON_BIN" scripts/align_hotword_manifest.py \
      --manifest "$SOURCE_MANIFEST" --catalog "$TARGET_CATALOG" \
      --output "$ALIGNED_MANIFEST" --report "$RUN_DIR/data/aligner_report.json" \
      --model "$ALIGNER_MODEL" --device cuda:0
  fi
  require_file "$ALIGNED_MANIFEST"
  for path in "$TIMED_MANIFEST" "$FOCUS_MANIFEST" "$RUN_DIR/data/preparation_report.json"; do
    protect_output "$path"
  done
  run_cmd "$PYTHON_BIN" scripts/prepare_online_manifest.py \
    --source-manifest "$SOURCE_MANIFEST" --aligned-manifest "$ALIGNED_MANIFEST" \
    --output "$TIMED_MANIFEST" --aligned-output "$FOCUS_MANIFEST" \
    --report "$RUN_DIR/data/preparation_report.json" "${overwrite_args[@]}"
}
stage1() {
  if [[ " $DATASETS " != *" boundary "* ]]; then return; fi
  echo '[stage1] build boundary WAVs with shifted entity timestamps'
  require_file "$FOCUS_MANIFEST"
  protect_output "$BOUNDARY_MANIFEST"
  protect_output "$BOUNDARY_WAV_DIR"
  run_cmd "$PYTHON_BIN" scripts/build_boundary_stress.py --manifest "$FOCUS_MANIFEST" \
    --output-dir "$BOUNDARY_WAV_DIR" --output-manifest "$BOUNDARY_MANIFEST" \
    --chunk-ms "$CHUNK_MS" "${overwrite_args[@]}"
}
stage2() {
  echo '[stage2] prepare the fixed catalog and checkpoint-matched indexes'
  if [[ ! -f "$CATALOG" ]]; then
    for path in "$HKUST_WORD_FREQ" "$MAGICDATA_WORD_FREQ" "$TARGET_CATALOG" "$FULL_MANIFEST"; do
      require_file "$path"
    done
    protect_output "$RUN_DIR/data/catalog_report.json"
    run_cmd "$PYTHON_BIN" scripts/build_glclap_evaluation_catalog.py \
      --word-freq "hkust=$HKUST_WORD_FREQ" --word-freq "magicdata=$MAGICDATA_WORD_FREQ" \
      --target-catalog "$TARGET_CATALOG" --eval-manifest "$FULL_MANIFEST" \
      --output "$CATALOG" --report "$RUN_DIR/data/catalog_report.json" \
      --size "$CATALOG_SIZE" --seed "$SEED" --version "aishell-ner-${SPLIT}-online-v2"
  fi
  require_file "$CATALOG"
  require_file "$CONFIG"
  for tag in "${checkpoint_tags[@]}"; do
    require_file "$EXP_DIR/$tag.pt"
    local index
    index="$(index_for "$tag")"
    if [[ -n "$INDEX_PATH" ]]; then
      require_file "$index"
      require_file "$index.json"
    else
      protect_output "$index"
      protect_output "$index.json"
      run_cmd "$PYTHON_BIN" scripts/build_glclap_index.py --config "$CONFIG" \
        --checkpoint "$EXP_DIR/$tag.pt" --catalog "$CATALOG" --output "$index" \
        --override "model.qwen_model=$QWEN_MODEL"
    fi
  done
}
stage3() {
  echo '[stage3] replay retrieval; fast is simulated FIFO, realtime is measured'
  require_file "$CATALOG"
  require_file "$CONFIG"
  local tag dataset replay index manifest output
  local -a verify_args=()
  if [[ "$VERIFY_OFFLINE" == 1 ]]; then verify_args+=(--verify-offline); fi
  for tag in "${checkpoint_tags[@]}"; do
    index="$(index_for "$tag")"
    require_file "$EXP_DIR/$tag.pt"
    require_file "$index"
    require_file "$index.json"
    for dataset in "${datasets[@]}"; do
      manifest="$(dataset_manifest "$dataset")"
      require_file "$manifest"
      for replay in "${replay_modes[@]}"; do
        output="$RUN_DIR/$tag/${dataset}_${replay}"
        protect_output "$output.jsonl"
        protect_output "$output.jsonl.run.json"
        protect_output "${output}_traces"
        run_cmd "$PYTHON_BIN" scripts/decode_streaming_retrieval.py \
          --config "$CONFIG" --checkpoint "$EXP_DIR/$tag.pt" --index "$index" \
          --manifest "$manifest" --output "$output.jsonl" --trace-dir "${output}_traces" \
          --mode streaming --replay-mode "$replay" --require-entity-timestamps \
          --require-index-metadata --expected-catalog "$CATALOG" --require-target-coverage \
          --warmup-refreshes "$WARMUP_REFRESHES" \
          --override "model.qwen_model=$QWEN_MODEL" \
          --override "streaming.chunk_size_sec=$CHUNK_SIZE_SEC" \
          --override "streaming.feed_step_ms=$FEED_STEP_MS" \
          --override "streaming.top_k=$TOP_K" --override "training.seed=$SEED" "${verify_args[@]}"
      done
    done
  done
}
stage4() {
  echo '[stage4] timely recall, latency components and paired bootstrap'
  local tag dataset replay output
  for tag in "${checkpoint_tags[@]}"; do
    for dataset in "${datasets[@]}"; do
      for replay in "${replay_modes[@]}"; do
        output="$RUN_DIR/$tag/${dataset}_${replay}"
        require_file "$output.jsonl"
        protect_output "${output}_metrics.json"
        protect_output "${output}_entities.jsonl"
        protect_output "${output}_refreshes.jsonl"
        run_cmd "$PYTHON_BIN" scripts/eval_hotword_retrieval.py \
          --input "$output.jsonl" --online --deadlines-ms "${deadline_values[@]}" \
          --output "${output}_metrics.json" --entity-output "${output}_entities.jsonl" \
          --refresh-output "${output}_refreshes.jsonl" \
          --bootstrap-samples "$BOOTSTRAP_SAMPLES" --seed "$SEED"
      done
    done
  done
}

die() { printf '[error] %s\n' "$*" >&2; exit 2; }
[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
[[ "$SPLIT" == dev || "$SPLIT" == test ]] || die 'SPLIT must be dev or test'
read -r -a checkpoint_tags <<< "$CHECKPOINTS"
read -r -a replay_modes <<< "$REPLAY_MODES"
read -r -a datasets <<< "$DATASETS"
read -r -a deadline_values <<< "$DEADLINES_MS"
for variable in CHECKPOINTS REPLAY_MODES DATASETS DEADLINES_MS; do
  [[ -n "${!variable// /}" ]] || die "$variable cannot be empty"
done
for tag in "${checkpoint_tags[@]}"; do [[ "$tag" == best || "$tag" == last ]] || die "unknown checkpoint: $tag"; done
for replay in "${replay_modes[@]}"; do [[ "$replay" == fast || "$replay" == realtime ]] || die "unknown replay: $replay"; done
for dataset in "${datasets[@]}"; do [[ "$dataset" == original || "$dataset" == boundary ]] || die "unknown dataset: $dataset"; done
[[ -z "$INDEX_PATH" || "${#checkpoint_tags[@]}" == 1 ]] || die 'INDEX_PATH requires a single CHECKPOINTS value'
for variable in CHUNK_MS FEED_STEP_MS TOP_K CATALOG_SIZE BOOTSTRAP_SAMPLES; do
  [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
done
for variable in WARMUP_REFRESHES SEED; do
  [[ "${!variable}" =~ ^(0|[1-9][0-9]*)$ ]] || die "$variable must be a nonnegative integer"
done
for variable in RUN_ALIGNER VERIFY_OFFLINE OVERWRITE DRY_RUN; do
  [[ "${!variable}" == 0 || "${!variable}" == 1 ]] || die "$variable must be 0 or 1"
done
(( TOP_K >= 50 )) || die 'TOP_K must be at least 50 for K=1/5/10/20/50 evaluation'
for deadline in "${deadline_values[@]}"; do
  [[ "$deadline" =~ ^(0|[1-9][0-9]*)$ ]] || die 'DEADLINES_MS requires nonnegative integer milliseconds'
done
printf -v CHUNK_SIZE_SEC '%d.%03d' "$((CHUNK_MS / 1000))" "$((CHUNK_MS % 1000))"
overwrite_args=()
[[ "$OVERWRITE" == 0 ]] || overwrite_args=(--overwrite)
[[ "$#" -gt 0 ]] || set -- all
printf '[online] split=%s experiment=%s output=%s replay=%s\n' "$SPLIT" "$EXP_DIR" "$RUN_DIR" "$REPLAY_MODES"
for stage in "$@"; do
  case "$stage" in
    all) stage0; stage1; stage2; stage3; stage4;;
    0|stage0) stage0;;
    1|stage1) stage1;;
    2|stage2) stage2;;
    3|stage3) stage3;;
    4|stage4) stage4;;
    *) die "unknown stage: $stage";;
  esac
done
