#!/usr/bin/env bash
# End-to-end GLCLAP accumulated-audio hotword retrieval experiment runner.
#
# This script intentionally keeps datasets, checkpoints, catalogs, indexes, and
# results outside Git. Configure paths with environment variables, then run one
# or more numbered stages, for example:
#
#   HKUST_WORD_FREQ=/data/hkust/word_freq.txt \
#   MAGICDATA_WORD_FREQ=/data/magicdata/word_freq.txt \
#   AISHELL1_TRAIN_MANIFEST=/data/manifests/aishell1_train.jsonl \
#   AISHELL_NER_ANNOTATED_TRANSCRIPT=/data/AISHELL-NER/data/aishell_ner_transcript.test.txt \
#   AISHELL_NER_WAV_ROOT=/data/AISHELL-1/wav/test bash run.sh stage0 stage1
#   bash run_aligner.sh all
#   bash run.sh stage2 stage3 stage4 stage5 stage6 stage7 stage8 stage9
#
# Run a subset or resume from a stage:
#   bash run.sh stage0 stage1 stage2
#   bash run.sh 3 4 6 7 8

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/glclap}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${DATA_ROOT}/glclap_qwen_feature_cache}"

HKUST_WORD_FREQ="${HKUST_WORD_FREQ:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/hkust_wo_st/word_freq.txt}"
MAGICDATA_WORD_FREQ="${MAGICDATA_WORD_FREQ:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/magicdata/word_freq.txt}"
AISHELL1_TRAIN_MANIFEST="${AISHELL1_TRAIN_MANIFEST:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/aishell/aishell_train.jsonl}"
AISHELL1_DEV_MANIFEST="${AISHELL1_DEV_MANIFEST:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/aishell/aishell_dev.jsonl}"
AISHELL_NER_ANNOTATED_TRANSCRIPT="${AISHELL_NER_ANNOTATED_TRANSCRIPT:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/AISHELL-NER/data/aishell_ner_transcript.test.txt}"
AISHELL_NER_WAV_ROOT="${AISHELL_NER_WAV_ROOT:-/data/zhengjie/datasets/asr/aishell/data_aishell/wav/test}"
AISHELL_NER_DIR="${AISHELL_NER_DIR:-/data/zhengjie/research/SLAM-LLM/examples/asr_librispeech/datasets/AISHELL-NER}"
AISHELL_NER_TARGET_CATALOG="${AISHELL_NER_TARGET_CATALOG:-${AISHELL1_NE_TARGET_CATALOG:-${AISHELL_NER_DIR}/targets_test.jsonl}}"
AISHELL_NER_EVAL_MANIFEST="${AISHELL_NER_EVAL_MANIFEST:-${AISHELL_NER_DIR}/test.jsonl}"
AISHELL_NER_ENTITY_MANIFEST="${AISHELL_NER_ENTITY_MANIFEST:-${AISHELL1_NE_EVAL_MANIFEST:-${AISHELL_NER_DIR}/test_entities.jsonl}}"
AISHELL_NER_ALIGNED_MANIFEST="${AISHELL_NER_ALIGNED_MANIFEST:-${AISHELL1_NE_ALIGNED_MANIFEST:-${AISHELL_NER_DIR}/test_aligned.jsonl}}"
AISHELL_NER_PREPARATION_REPORT="${AISHELL_NER_PREPARATION_REPORT:-${AISHELL_NER_DIR}/test_preparation_report.json}"

# Backward-compatible aliases for pre-AISHELL-NER run commands.
AISHELL1_NE_TARGET_CATALOG="${AISHELL1_NE_TARGET_CATALOG:-${AISHELL_NER_TARGET_CATALOG}}"
AISHELL1_NE_EVAL_MANIFEST="${AISHELL1_NE_EVAL_MANIFEST:-${AISHELL_NER_ENTITY_MANIFEST}}"
AISHELL1_NE_ALIGNED_MANIFEST="${AISHELL1_NE_ALIGNED_MANIFEST:-${AISHELL_NER_ALIGNED_MANIFEST}}"

HOTWORD_DIR="${HOTWORD_DIR:-${DATA_ROOT}/hotwords}"
NEGATIVE_CATALOG="${NEGATIVE_CATALOG:-${HOTWORD_DIR}/zh_train_10k.jsonl}"
EVALUATION_CATALOG="${EVALUATION_CATALOG:-${HOTWORD_DIR}/aishell_ner_10k.jsonl}"
TRAINING_POOL_REPORT="${TRAINING_POOL_REPORT:-${HOTWORD_DIR}/training_pool_report.json}"
EVALUATION_CATALOG_REPORT="${EVALUATION_CATALOG_REPORT:-${CATALOG_REPORT:-${HOTWORD_DIR}/evaluation_catalog_report.json}}"
CATALOG_REPORT="${CATALOG_REPORT:-${EVALUATION_CATALOG_REPORT}}"
BOUNDARY_WAV_DIR="${BOUNDARY_WAV_DIR:-${AISHELL_NER_DIR}/boundary_wav}"
BOUNDARY_MANIFEST="${BOUNDARY_MANIFEST:-${AISHELL_NER_DIR}/test_boundary.jsonl}"

QWEN_MODEL="${QWEN_MODEL:-/data/zhengjie/resources/pretrain_models/Qwen3-ASR-0.6B}"
CATALOG_SIZE="${CATALOG_SIZE:-10000}"
SEED="${SEED:-42}"
CHUNK_MS="${CHUNK_MS:-2000}"
CHUNK_SIZE_SEC="${CHUNK_SIZE_SEC:-2.0}"
FEED_STEP_MS="${FEED_STEP_MS:-100}"
TOP_K="${TOP_K:-50}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-384}"
AUDIO_BATCHING="${AUDIO_BATCHING:-packed}"
AUDIO_LOADER_WORKERS="${AUDIO_LOADER_WORKERS:-4}"
TEXT_CACHE_MAX_ENTRIES="${TEXT_CACHE_MAX_ENTRIES:-20000}"
VISIBLE_GPU_COUNT=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#VISIBLE_GPU_IDS[@]}"
fi
NUM_GPUS="${NUM_GPUS:-${VISIBLE_GPU_COUNT}}"

# Stage switches. Each training variant uses torchrun when NUM_GPUS > 1.
INSTALL_DEPS="${INSTALL_DEPS:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
RESUME_TRAINING="${RESUME_TRAINING:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
VERIFY_OFFLINE="${VERIFY_OFFLINE:-1}"
RUN_STRESS="${RUN_STRESS:-1}"

GLOBAL_ONLY_CONFIG="configs/glclap/global_only_clap.yaml"
FROZEN_CONFIG="configs/glclap/qwen_post_projector_frozen.yaml"
WARMSTART_CONFIG="configs/glclap/qwen_projector_warmstart.yaml"
RANDOM_CONFIG="configs/glclap/random_projector.yaml"

mkdir -p "${LOG_ROOT}"
RUN_LOG="${LOG_ROOT}/run-$(date '+%Y%m%d-%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

trap 'status=$?; echo "[error] command failed at ${BASH_SOURCE[0]}:${LINENO} (exit=${status})"; exit "${status}"' ERR

usage() {
  cat <<'EOF'
Usage: bash run.sh <all|stage0|stage1|...|stage9> [...]

Stages:
  stage0  Environment, pinned dependency, CUDA, source, and metadata checks
  stage1  Build training pool, parse AISHELL-NER gold labels, and build evaluation catalog
  stage2  Build Center/B-400/B-200/B-100/Cross-25/50/75 boundary WAVs
  stage3  Train the global-only CLAP baseline
  stage4  Train the frozen Qwen projector GLCLAP main system
  stage5  Train warm-start and random-projector initialization ablations
  stage6  Build an fp16 text embedding index for every trained variant
  stage7  Run accumulated-audio streaming retrieval on the boundary manifest
  stage8  Evaluate retrieval, latency, boundary penalty, parity, and bootstrap CI
  stage9  Run unit/stress tests and print the experiment artifact summary

Important environment variables:
  HKUST_WORD_FREQ, MAGICDATA_WORD_FREQ
  AISHELL1_TRAIN_MANIFEST, AISHELL1_DEV_MANIFEST
  AISHELL_NER_ANNOTATED_TRANSCRIPT, AISHELL_NER_WAV_ROOT
  AISHELL_NER_TARGET_CATALOG, AISHELL_NER_EVAL_MANIFEST
  AISHELL_NER_ENTITY_MANIFEST, AISHELL_NER_ALIGNED_MANIFEST
  QWEN_MODEL, DATA_ROOT, OUTPUT_ROOT, CUDA_VISIBLE_DEVICES
  EVAL_STRATEGY=epoch|steps, EVAL_STEPS=100, EVAL_MAX_SAMPLES=0, EVAL_BATCH_SIZE=8
  NUM_GPUS=<visible GPU count>, GLOBAL_BATCH_SIZE=384
  FEATURE_CACHE_DIR=<shared fast disk>, AUDIO_BATCHING=packed|serial
  AUDIO_LOADER_WORKERS=4, TEXT_CACHE_MAX_ENTRIES=20000
  INSTALL_DEPS=1, RUN_ABLATIONS=0, VERIFY_OFFLINE=0, RUN_STRESS=0
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    *) echo "invalid boolean value: $1" >&2; return 2 ;;
  esac
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[missing] ${label}: ${path}" >&2
    exit 2
  fi
}

require_directory() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[missing] ${label}: ${path}" >&2
    exit 2
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[missing] required command: ${command_name}" >&2
    exit 2
  fi
}

variant_config() {
  case "$1" in
    global_only) echo "${GLOBAL_ONLY_CONFIG}" ;;
    frozen) echo "${FROZEN_CONFIG}" ;;
    warmstart) echo "${WARMSTART_CONFIG}" ;;
    random) echo "${RANDOM_CONFIG}" ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

variant_dir() {
  echo "${OUTPUT_ROOT}/$1"
}

active_variants() {
  echo global_only
  echo frozen
  if is_true "${RUN_ABLATIONS}"; then
    echo warmstart
    echo random
  fi
}

train_variant() {
  local variant="$1"
  local config
  local output_dir
  local checkpoint
  local -a command_args
  local -a launcher_args
  config="$(variant_config "${variant}")"
  output_dir="$(variant_dir "${variant}")"
  checkpoint="${output_dir}/last.pt"
  require_file "${config}" "${variant} config"
  require_file "${AISHELL1_TRAIN_MANIFEST}" "AISHELL-1 training manifest"
  if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
    return 2
  fi
  require_file "${AISHELL1_DEV_MANIFEST}" "AISHELL-1 validation manifest"
  require_file "${NEGATIVE_CATALOG}" "training negative catalog"
  mkdir -p "${output_dir}"
  command_args=(
    scripts/train_glclap_retriever.py
    --config "${config}"
    --manifest "${AISHELL1_TRAIN_MANIFEST}"
    --dev-manifest "${AISHELL1_DEV_MANIFEST}"
    --negative-catalog "${NEGATIVE_CATALOG}"
    --output-dir "${output_dir}"
    --feature-cache-dir "${FEATURE_CACHE_DIR}"
    --override "model.qwen_model=${QWEN_MODEL}"
    --override "training.seed=${SEED}"
    --override "training.global_batch_size=${GLOBAL_BATCH_SIZE}"
    --override "training.audio_batching=${AUDIO_BATCHING}"
    --override "training.audio_loader_workers=${AUDIO_LOADER_WORKERS}"
    --override "training.text_cache_max_entries=${TEXT_CACHE_MAX_ENTRIES}"
    --override "evaluation.strategy=${EVAL_STRATEGY}"
    --override "evaluation.steps=${EVAL_STEPS}"
    --override "evaluation.max_samples=${EVAL_MAX_SAMPLES}"
    --override "evaluation.batch_size=${EVAL_BATCH_SIZE}"
  )
  if is_true "${RESUME_TRAINING}" && [[ -f "${checkpoint}" ]]; then
    command_args+=(--resume "${checkpoint}")
  fi
  launcher_args=("${PYTHON_BIN}")
  if (( NUM_GPUS > 1 )); then
    launcher_args=("${PYTHON_BIN}" -m torch.distributed.run --standalone "--nproc_per_node=${NUM_GPUS}")
  fi
  echo "[train] variant=${variant} num_gpus=${NUM_GPUS} global_batch=${GLOBAL_BATCH_SIZE} audio_batching=${AUDIO_BATCHING}"
  echo "[train] feature_cache=${FEATURE_CACHE_DIR} audio_workers=${AUDIO_LOADER_WORKERS}"
  "${launcher_args[@]}" "${command_args[@]}"
  require_file "${checkpoint}" "${variant} final checkpoint"
  require_file "${output_dir}/best.pt" "${variant} best validation checkpoint"
}

stage0() {
  echo "[stage0] environment and reproducibility checks"
  require_command "${PYTHON_BIN}"
  require_command git
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[warning] official Qwen/vLLM runs are validated on Linux CUDA, current OS: $(uname -s)"
  fi
  if is_true "${INSTALL_DEPS}"; then
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -e '.[probe,qwen,config]'
  fi
  "${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
  "${PYTHON_BIN}" -c 'import importlib.metadata as m; expected={"qwen-asr":"0.0.6","transformers":"4.57.6","vllm":"0.14.0"}; actual={k:m.version(k) for k in expected}; wrong={k:(actual[k],v) for k,v in expected.items() if actual[k]!=v}; assert not wrong, wrong; print(actual)'
  if is_true "${REQUIRE_CUDA}"; then
    "${PYTHON_BIN}" -c "import torch; requested=int('${NUM_GPUS}'); assert torch.cuda.is_available(), 'CUDA is required'; assert torch.cuda.device_count() >= requested, (torch.cuda.device_count(), requested); print({'cuda':torch.version.cuda,'devices':torch.cuda.device_count(),'requested_training_gpus':requested,'device0':torch.cuda.get_device_name(0)})"
  fi
  "${PYTHON_BIN}" -c "chunk_ms=float('${CHUNK_MS}'); chunk_sec=float('${CHUNK_SIZE_SEC}'); assert abs(chunk_ms/1000.0-chunk_sec)<1e-9, (chunk_ms,chunk_sec)"
  "${PYTHON_BIN}" -m compileall -q asr scripts tests
  mkdir -p "${OUTPUT_ROOT}/run_metadata"
  git rev-parse HEAD | tee "${OUTPUT_ROOT}/run_metadata/git_commit.txt"
  "${PYTHON_BIN}" -m pip freeze | tee "${OUTPUT_ROOT}/run_metadata/environment.freeze.txt"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee "${OUTPUT_ROOT}/run_metadata/nvidia-smi.txt"
  fi
}

stage1() {
  echo "[stage1a] build training-negative pool without evaluation annotations"
  require_file "${HKUST_WORD_FREQ}" "HKUST word frequency file"
  require_file "${MAGICDATA_WORD_FREQ}" "MagicData word frequency file"
  mkdir -p "${HOTWORD_DIR}"
  "${PYTHON_BIN}" scripts/build_glclap_training_pool.py \
    --word-freq "hkust=${HKUST_WORD_FREQ}" \
    --word-freq "magicdata=${MAGICDATA_WORD_FREQ}" \
    --output "${NEGATIVE_CATALOG}" \
    --report "${TRAINING_POOL_REPORT}" \
    --size "${CATALOG_SIZE}" \
    --seed "${SEED}"
  require_file "${NEGATIVE_CATALOG}" "generated training negative pool"
  require_file "${TRAINING_POOL_REPORT}" "training pool report"

  echo "[stage1b] parse official AISHELL-NER gold entity annotations"
  require_file "${AISHELL_NER_ANNOTATED_TRANSCRIPT}" "AISHELL-NER tagged transcript"
  require_directory "${AISHELL_NER_WAV_ROOT}" "AISHELL-1 test WAV root"
  mkdir -p "${AISHELL_NER_DIR}"
  "${PYTHON_BIN}" scripts/prepare_aishell_ner.py \
    --annotated-transcript "${AISHELL_NER_ANNOTATED_TRANSCRIPT}" \
    --wav-root "${AISHELL_NER_WAV_ROOT}" \
    --target-catalog-output "${AISHELL_NER_TARGET_CATALOG}" \
    --eval-manifest-output "${AISHELL_NER_EVAL_MANIFEST}" \
    --entity-manifest-output "${AISHELL_NER_ENTITY_MANIFEST}" \
    --report "${AISHELL_NER_PREPARATION_REPORT}" \
    --split test
  require_file "${AISHELL_NER_TARGET_CATALOG}" "AISHELL-NER gold target catalog"
  require_file "${AISHELL_NER_EVAL_MANIFEST}" "AISHELL-NER full evaluation manifest"
  require_file "${AISHELL_NER_ENTITY_MANIFEST}" "AISHELL-NER entity-only manifest"
  require_file "${AISHELL_NER_PREPARATION_REPORT}" "AISHELL-NER preparation report"

  echo "[stage1c] build evaluation target+distractor catalog"
  "${PYTHON_BIN}" scripts/build_glclap_evaluation_catalog.py \
    --word-freq "hkust=${HKUST_WORD_FREQ}" \
    --word-freq "magicdata=${MAGICDATA_WORD_FREQ}" \
    --target-catalog "${AISHELL_NER_TARGET_CATALOG}" \
    --eval-manifest "${AISHELL_NER_EVAL_MANIFEST}" \
    --output "${EVALUATION_CATALOG}" \
    --report "${EVALUATION_CATALOG_REPORT}" \
    --size "${CATALOG_SIZE}" \
    --seed "${SEED}"
  require_file "${EVALUATION_CATALOG}" "generated evaluation catalog"
  require_file "${EVALUATION_CATALOG_REPORT}" "evaluation catalog report"
}

stage2() {
  echo "[stage2] build controlled chunk-boundary evaluation audio"
  if [[ ! -f "${AISHELL_NER_ALIGNED_MANIFEST}" ]]; then
    echo "[missing] aligned AISHELL-NER manifest: ${AISHELL_NER_ALIGNED_MANIFEST}" >&2
    echo "[hint] create it first with: bash run_aligner.sh all" >&2
    exit 2
  fi
  mkdir -p "${BOUNDARY_WAV_DIR}" "$(dirname -- "${BOUNDARY_MANIFEST}")"
  "${PYTHON_BIN}" scripts/build_boundary_stress.py \
    --manifest "${AISHELL_NER_ALIGNED_MANIFEST}" \
    --output-dir "${BOUNDARY_WAV_DIR}" \
    --output-manifest "${BOUNDARY_MANIFEST}" \
    --chunk-ms "${CHUNK_MS}"
  require_file "${BOUNDARY_MANIFEST}" "generated boundary manifest"
}

stage3() {
  echo "[stage3] train global-only CLAP baseline"
  train_variant global_only
}

stage4() {
  echo "[stage4] train frozen Qwen-projector GLCLAP main system"
  train_variant frozen
}

stage5() {
  echo "[stage5] train initialization ablations"
  if ! is_true "${RUN_ABLATIONS}"; then
    echo "[stage5] skipped because RUN_ABLATIONS=${RUN_ABLATIONS}"
    return
  fi
  train_variant warmstart
  train_variant random
}

stage6() {
  echo "[stage6] build exact fp16 embedding indexes"
  require_file "${EVALUATION_CATALOG}" "evaluation catalog"
  local variant
  local config
  local output_dir
  local checkpoint
  local index_path
  while IFS= read -r variant; do
    config="$(variant_config "${variant}")"
    output_dir="$(variant_dir "${variant}")"
    checkpoint="${output_dir}/best.pt"
    index_path="${output_dir}/aishell_ner_${CATALOG_SIZE}.npz"
    require_file "${checkpoint}" "${variant} checkpoint"
    mkdir -p "${output_dir}"
    "${PYTHON_BIN}" scripts/build_glclap_index.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --catalog "${EVALUATION_CATALOG}" \
      --output "${index_path}" \
      --override "model.qwen_model=${QWEN_MODEL}"
    require_file "${index_path}" "${variant} embedding index"
    require_file "${index_path}.json" "${variant} index metadata"
  done < <(active_variants)
}

stage7() {
  echo "[stage7] accumulated-audio streaming retrieval"
  require_file "${BOUNDARY_MANIFEST}" "boundary evaluation manifest"
  local variant
  local config
  local output_dir
  local checkpoint
  local index_path
  local -a verify_args
  verify_args=()
  if is_true "${VERIFY_OFFLINE}"; then
    verify_args+=(--verify-offline)
  fi
  while IFS= read -r variant; do
    config="$(variant_config "${variant}")"
    output_dir="$(variant_dir "${variant}")"
    checkpoint="${output_dir}/best.pt"
    index_path="${output_dir}/aishell_ner_${CATALOG_SIZE}.npz"
    require_file "${checkpoint}" "${variant} checkpoint"
    require_file "${index_path}" "${variant} embedding index"
    "${PYTHON_BIN}" scripts/decode_streaming_retrieval.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --index "${index_path}" \
      --manifest "${BOUNDARY_MANIFEST}" \
      --output "${output_dir}/boundary_retrieval.jsonl" \
      --trace-dir "${output_dir}/traces" \
      --override "model.qwen_model=${QWEN_MODEL}" \
      --override "streaming.chunk_size_sec=${CHUNK_SIZE_SEC}" \
      --override "streaming.feed_step_ms=${FEED_STEP_MS}" \
      --override "streaming.top_k=${TOP_K}" \
      "${verify_args[@]}"
    require_file "${output_dir}/boundary_retrieval.jsonl" "${variant} retrieval output"
  done < <(active_variants)
}

stage8() {
  echo "[stage8] evaluate retrieval and paired gains"
  local baseline_output="${OUTPUT_ROOT}/global_only/boundary_retrieval.jsonl"
  local variant
  local output_dir
  require_file "${baseline_output}" "global-only retrieval output"
  "${PYTHON_BIN}" scripts/eval_hotword_retrieval.py \
    --input "${baseline_output}" \
    --output "${OUTPUT_ROOT}/global_only/metrics.json" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --seed "${SEED}"
  while IFS= read -r variant; do
    if [[ "${variant}" == "global_only" ]]; then
      continue
    fi
    output_dir="$(variant_dir "${variant}")"
    require_file "${output_dir}/boundary_retrieval.jsonl" "${variant} retrieval output"
    "${PYTHON_BIN}" scripts/eval_hotword_retrieval.py \
      --input "${output_dir}/boundary_retrieval.jsonl" \
      --baseline "${baseline_output}" \
      --output "${output_dir}/metrics.json" \
      --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
      --seed "${SEED}"
  done < <(active_variants)
}

stage9() {
  echo "[stage9] repository tests, stress checks, and artifact summary"
  "${PYTHON_BIN}" -m compileall -q asr scripts tests
  "${PYTHON_BIN}" -m unittest discover -s tests -v
  if is_true "${RUN_STRESS}"; then
    RUN_CONTEXTUAL_STRESS=1 "${PYTHON_BIN}" -m unittest tests.test_contextual_stress -v
    RUN_GLCLAP_STRESS=1 "${PYTHON_BIN}" -m unittest tests.test_glclap_stress -v
  fi
  echo "[summary] AISHELL-NER gold views"
  printf '  %s\n' "${AISHELL_NER_TARGET_CATALOG}" "${AISHELL_NER_EVAL_MANIFEST}" "${AISHELL_NER_ENTITY_MANIFEST}"
  printf '  %s\n' "${AISHELL_NER_PREPARATION_REPORT}"
  echo "[summary] catalogs"
  printf '  %s\n' "${NEGATIVE_CATALOG}" "${TRAINING_POOL_REPORT}"
  printf '  %s\n' "${EVALUATION_CATALOG}" "${EVALUATION_CATALOG_REPORT}"
  echo "[summary] boundary manifest"
  printf '  %s\n' "${BOUNDARY_MANIFEST}"
  echo "[summary] metrics"
  local variant
  while IFS= read -r variant; do
    printf '  %s\n' "$(variant_dir "${variant}")/metrics.json"
  done < <(active_variants)
  echo "[summary] run log: ${RUN_LOG}"
}

run_stage() {
  local requested="$1"
  case "${requested}" in
    0|stage0) stage0 ;;
    1|stage1) stage1 ;;
    2|stage2) stage2 ;;
    3|stage3) stage3 ;;
    4|stage4) stage4 ;;
    5|stage5) stage5 ;;
    6|stage6) stage6 ;;
    7|stage7) stage7 ;;
    8|stage8) stage8 ;;
    9|stage9) stage9 ;;
    *) echo "unknown stage: ${requested}" >&2; usage; exit 2 ;;
  esac
}

main() {
  if [[ "$#" -eq 0 ]]; then
    usage
    exit 2
  fi
  echo "[run] repository=${REPO_ROOT}"
  echo "[run] output_root=${OUTPUT_ROOT}"
  echo "[run] qwen_model=${QWEN_MODEL}"
  echo "[run] stages=$*"
  if [[ "$1" == "all" ]]; then
    if [[ "$#" -ne 1 ]]; then
      echo "'all' cannot be combined with explicit stages" >&2
      exit 2
    fi
    local stage_number
    for stage_number in {0..9}; do
      run_stage "stage${stage_number}"
    done
  else
    local requested
    for requested in "$@"; do
      run_stage "${requested}"
    done
  fi
  echo "[done] requested stages completed"
}

main "$@"
