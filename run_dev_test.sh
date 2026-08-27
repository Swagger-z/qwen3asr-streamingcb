#!/usr/bin/env bash
# Formal, fixed-catalog DEV retrieval; no training, aligner or boundary expansion.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_DIR="${EXP_DIR:-${REPO_ROOT}/outputs/glclap/frozen_bs8_100epoch}"
RUN_DIR="${RUN_DIR:-${EXP_DIR}/dev_formal}"
CONFIG="${CONFIG:-configs/glclap/qwen_post_projector_frozen.yaml}"
QWEN_MODEL="${QWEN_MODEL:-/data/zhengjie/resources/pretrain_models/Qwen3-ASR-0.6B}"
DATA_ROOT="${DATA_ROOT:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets}"
AISHELL_NER_DIR="${AISHELL_NER_DIR:-${DATA_ROOT}/AISHELL-NER}"
DEV_ANNOTATED_TRANSCRIPT="${DEV_ANNOTATED_TRANSCRIPT:-${AISHELL_NER_DIR}/data/aishell_ner_transcript.dev.txt}"
DEV_WAV_ROOT="${DEV_WAV_ROOT:-/data/zhengjie/datasets/asr/aishell/data_aishell/wav/dev}"
DEV_TARGET_CATALOG="${DEV_TARGET_CATALOG:-${RUN_DIR}/data/targets_dev.jsonl}"
DEV_FULL_MANIFEST="${DEV_FULL_MANIFEST:-${RUN_DIR}/data/dev.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-${RUN_DIR}/data/dev_entities.jsonl}"
PREPARATION_REPORT="${PREPARATION_REPORT:-${RUN_DIR}/data/dev_preparation_report.json}"
HKUST_WORD_FREQ="${HKUST_WORD_FREQ:-${DATA_ROOT}/hkust_wo_st/word_freq.txt}"
MAGICDATA_WORD_FREQ="${MAGICDATA_WORD_FREQ:-${DATA_ROOT}/magicdata/word_freq.txt}"
CATALOG_SIZE="${CATALOG_SIZE:-10000}"
CATALOG="${CATALOG:-${RUN_DIR}/data/aishell_ner_dev_${CATALOG_SIZE}.jsonl}"
CATALOG_REPORT="${CATALOG_REPORT:-${RUN_DIR}/data/catalog_report.json}"
CHECKPOINTS="${CHECKPOINTS:-best last}"
MODES="${MODES:-offline streaming}"
SEED="${SEED:-42}"
TOP_K="${TOP_K:-50}"
CHUNK_SIZE_SEC="${CHUNK_SIZE_SEC:-2.0}"
FEED_STEP_MS="${FEED_STEP_MS:-100}"
VERIFY_OFFLINE="${VERIFY_OFFLINE:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'HELP'
Usage: bash run_dev_test.sh [all | stage0 stage1 stage2 stage3 stage4]
  stage0  Prepare gold DEV catalog + full/entity-only manifests (no aligner).
  stage1  Build fixed DEV 10k catalog with the same builder as test.
  stage2  Build a separate DEV index for each selected checkpoint.
  stage3  Run whole-audio offline and accumulated-audio streaming retrieval.
  stage4  Evaluate Hit@K, entity Recall@K, MRR and timing with the test evaluator.

Defaults: CHECKPOINTS="best last", MODES="offline streaming", TOP_K=50.
EXP_DIR selects a training directory containing best.pt / last.pt.
RUN_DIR defaults to EXP_DIR/dev_formal. Existing files are protected;
OVERWRITE=1 explicitly permits replacing outputs. Use a new RUN_DIR for a new run.
DRY_RUN=1 prints commands without loading models, requiring data or writing files.
Use CUDA_VISIBLE_DEVICES=4 to select one GPU; this script does not use DDP.
To reuse prepared DEV files, set DEV_MANIFEST, DEV_FULL_MANIFEST,
DEV_TARGET_CATALOG and start at stage1. See docs/glclap_dev_formal_test.md.
HELP
}

run_cmd() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

require_file() {
  if [[ "${DRY_RUN}" != 1 && ! -f "$1" ]]; then
    echo "[missing] $1 (run the prerequisite stage or override its path)" >&2
    exit 2
  fi
}

protect_output() {
  if [[ "${DRY_RUN}" != 1 && "${OVERWRITE}" != 1 && -e "$1" ]]; then
    echo "[exists] $1; select a new RUN_DIR or set OVERWRITE=1" >&2
    exit 2
  fi
}

stage0() {
  echo '[stage0] prepare original DEV audio and all gold entities per utterance'
  require_file "${DEV_ANNOTATED_TRANSCRIPT}"
  if [[ "${DRY_RUN}" != 1 && ! -d "${DEV_WAV_ROOT}" ]]; then
    echo "[missing] DEV_WAV_ROOT=${DEV_WAV_ROOT}" >&2; exit 2
  fi
  for path in "${DEV_TARGET_CATALOG}" "${DEV_FULL_MANIFEST}" "${DEV_MANIFEST}" "${PREPARATION_REPORT}"; do
    protect_output "${path}"
  done
  run_cmd "${PYTHON_BIN}" scripts/prepare_aishell_ner.py \
    --annotated-transcript "${DEV_ANNOTATED_TRANSCRIPT}" --wav-root "${DEV_WAV_ROOT}" \
    --target-catalog-output "${DEV_TARGET_CATALOG}" \
    --eval-manifest-output "${DEV_FULL_MANIFEST}" --entity-manifest-output "${DEV_MANIFEST}" \
    --report "${PREPARATION_REPORT}" --split dev
}

stage1() {
  echo '[stage1] build fixed DEV target+distractor catalog (not the training pool)'
  for path in "${HKUST_WORD_FREQ}" "${MAGICDATA_WORD_FREQ}" "${DEV_TARGET_CATALOG}" "${DEV_FULL_MANIFEST}"; do
    require_file "${path}"
  done
  protect_output "${CATALOG}"
  protect_output "${CATALOG_REPORT}"
  run_cmd "${PYTHON_BIN}" scripts/build_glclap_evaluation_catalog.py \
    --word-freq "hkust=${HKUST_WORD_FREQ}" --word-freq "magicdata=${MAGICDATA_WORD_FREQ}" \
    --target-catalog "${DEV_TARGET_CATALOG}" --eval-manifest "${DEV_FULL_MANIFEST}" \
    --output "${CATALOG}" --report "${CATALOG_REPORT}" --size "${CATALOG_SIZE}" \
    --seed "${SEED}" --version "aishell-ner-dev-${CATALOG_SIZE}-v1"
}

stage2() {
  echo '[stage2] build checkpoint-specific DEV text indexes'
  require_file "${CATALOG}"
  require_file "${CONFIG}"
  local tag index_path
  for tag in "${checkpoint_tags[@]}"; do
    require_file "${EXP_DIR}/${tag}.pt"
    index_path="${RUN_DIR}/${tag}/aishell_ner_dev_${CATALOG_SIZE}.npz"
    protect_output "${index_path}"
    protect_output "${index_path}.json"
    run_cmd "${PYTHON_BIN}" scripts/build_glclap_index.py \
      --config "${CONFIG}" --checkpoint "${EXP_DIR}/${tag}.pt" \
      --catalog "${CATALOG}" --output "${index_path}" \
      --override "model.qwen_model=${QWEN_MODEL}" --override "training.seed=${SEED}"
  done
}

stage3() {
  echo '[stage3] formal DEV retrieval with whole-utterance gold sets'
  require_file "${DEV_MANIFEST}"
  require_file "${CATALOG}"
  require_file "${CONFIG}"
  local tag mode index_path output_dir
  local -a verify_args
  for tag in "${checkpoint_tags[@]}"; do
    index_path="${RUN_DIR}/${tag}/aishell_ner_dev_${CATALOG_SIZE}.npz"
    require_file "${EXP_DIR}/${tag}.pt"
    require_file "${index_path}"
    require_file "${index_path}.json"
    for mode in "${retrieval_modes[@]}"; do
      output_dir="${RUN_DIR}/${tag}"
      protect_output "${output_dir}/${mode}.jsonl"
      protect_output "${output_dir}/${mode}.jsonl.run.json"
      protect_output "${output_dir}/traces_${mode}"
      verify_args=()
      if [[ "${mode}" == streaming && "${VERIFY_OFFLINE}" == 1 ]]; then
        verify_args+=(--verify-offline)
      fi
      run_cmd "${PYTHON_BIN}" scripts/decode_streaming_retrieval.py \
        --config "${CONFIG}" --checkpoint "${EXP_DIR}/${tag}.pt" --index "${index_path}" \
        --manifest "${DEV_MANIFEST}" --mode "${mode}" \
        --output "${output_dir}/${mode}.jsonl" --trace-dir "${output_dir}/traces_${mode}" \
        --require-index-metadata --require-target-coverage --expected-catalog "${CATALOG}" \
        --override "model.qwen_model=${QWEN_MODEL}" --override "training.seed=${SEED}" \
        --override "streaming.chunk_size_sec=${CHUNK_SIZE_SEC}" \
        --override "streaming.feed_step_ms=${FEED_STEP_MS}" --override "streaming.top_k=${TOP_K}" \
        "${verify_args[@]}"
    done
  done
}

stage4() {
  echo '[stage4] same evaluator as test: Hit@K and entity Recall@K are separate'
  local tag mode
  for tag in "${checkpoint_tags[@]}"; do
    for mode in "${retrieval_modes[@]}"; do
      require_file "${RUN_DIR}/${tag}/${mode}.jsonl"
      protect_output "${RUN_DIR}/${tag}/${mode}_metrics.json"
      run_cmd "${PYTHON_BIN}" scripts/eval_hotword_retrieval.py \
        --input "${RUN_DIR}/${tag}/${mode}.jsonl" \
        --output "${RUN_DIR}/${tag}/${mode}_metrics.json" --seed "${SEED}"
    done
  done
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
read -r -a checkpoint_tags <<< "${CHECKPOINTS}"
read -r -a retrieval_modes <<< "${MODES}"
if (( ${#checkpoint_tags[@]} == 0 || ${#retrieval_modes[@]} == 0 )); then
  echo 'CHECKPOINTS and MODES must not be empty' >&2; exit 2
fi
for tag in "${checkpoint_tags[@]}"; do
  case "${tag}" in best|last) ;; *) echo "invalid checkpoint: ${tag}" >&2; exit 2 ;; esac
done
for mode in "${retrieval_modes[@]}"; do
  case "${mode}" in offline|streaming) ;; *) echo "invalid mode: ${mode}" >&2; exit 2 ;; esac
done
if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]] || (( TOP_K < 50 )); then
  echo 'TOP_K must be >= 50 because the evaluator reports Recall@50' >&2; exit 2
fi
stages=("$@")
if (( $# == 0 )) || [[ "$*" == all ]]; then stages=(stage0 stage1 stage2 stage3 stage4); fi
for stage in "${stages[@]}"; do
  case "${stage}" in 0|1|2|3|4|stage0|stage1|stage2|stage3|stage4) ;; *) usage; exit 2 ;; esac
done
echo "[dev-test] experiment=${EXP_DIR} results=${RUN_DIR} checkpoints=${CHECKPOINTS} modes=${MODES}"
for stage in "${stages[@]}"; do
  case "${stage}" in
    0|stage0) stage0 ;; 1|stage1) stage1 ;; 2|stage2) stage2 ;; 3|stage3) stage3 ;; 4|stage4) stage4 ;;
  esac
done
