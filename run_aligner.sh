#!/usr/bin/env bash
# Offline Qwen3-ForcedAligner preparation for the stage2 boundary manifest.

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/glclap}"
AISHELL_NER_DIR="${AISHELL_NER_DIR:-${DATA_ROOT}/aishell_ner}"
AISHELL_NER_ENTITY_MANIFEST="${AISHELL_NER_ENTITY_MANIFEST:-${AISHELL1_NE_EVAL_MANIFEST:-${AISHELL_NER_DIR}/test_entities.jsonl}}"
AISHELL_NER_TARGET_CATALOG="${AISHELL_NER_TARGET_CATALOG:-${AISHELL1_NE_TARGET_CATALOG:-${AISHELL_NER_DIR}/targets_test.jsonl}}"
AISHELL_NER_ALIGNED_MANIFEST="${AISHELL_NER_ALIGNED_MANIFEST:-${AISHELL1_NE_ALIGNED_MANIFEST:-${AISHELL_NER_DIR}/test_aligned.jsonl}}"
ALIGNMENT_REPORT="${ALIGNMENT_REPORT:-${OUTPUT_ROOT}/alignment/qwen3_alignment_report.json}"
ALIGNMENT_VALIDATION_REPORT="${ALIGNMENT_VALIDATION_REPORT:-${OUTPUT_ROOT}/alignment/validation_report.json}"
ALIGNMENT_TRACE="${ALIGNMENT_TRACE:-${OUTPUT_ROOT}/alignment/qwen3_alignment.trace.jsonl}"

ALIGNER_MODEL="${ALIGNER_MODEL:-Qwen/Qwen3-ForcedAligner-0.6B}"
ALIGNER_DEVICE="${ALIGNER_DEVICE:-cuda:0}"
ALIGNER_DTYPE="${ALIGNER_DTYPE:-bfloat16}"
ALIGNER_BATCH_SIZE="${ALIGNER_BATCH_SIZE:-8}"
ALIGNER_LANGUAGE="${ALIGNER_LANGUAGE:-Chinese}"
ALIGNER_ATTN_IMPLEMENTATION="${ALIGNER_ATTN_IMPLEMENTATION:-}"
OCCURRENCE_POLICY="${OCCURRENCE_POLICY:-error}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
RESUME_ALIGNMENT="${RESUME_ALIGNMENT:-1}"
OVERWRITE_ALIGNMENT="${OVERWRITE_ALIGNMENT:-0}"
ON_ALIGNMENT_ERROR="${ON_ALIGNMENT_ERROR:-fail}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"

LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
mkdir -p "${LOG_ROOT}"
RUN_LOG="${LOG_ROOT}/aligner-$(date '+%Y%m%d-%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

trap 'status=$?; echo "[error] command failed at ${BASH_SOURCE[0]}:${LINENO} (exit=${status})"; exit "${status}"' ERR

usage() {
  cat <<'EOF'
Usage: bash run_aligner.sh <all|stage0|stage1|stage2> [...]

  stage0  Check inputs, pinned qwen-asr runtime, and CUDA
  stage1  Run Qwen3-ForcedAligner and create test_aligned.jsonl
  stage2  Validate completeness, timestamps, and PCM16 WAV compatibility

Required inputs:
  AISHELL_NER_ENTITY_MANIFEST
  AISHELL_NER_TARGET_CATALOG

Primary output:
  AISHELL_NER_ALIGNED_MANIFEST
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
  if [[ ! -f "$1" ]]; then
    echo "[missing] $2: $1" >&2
    exit 2
  fi
}

stage0() {
  echo "[stage0] check forced-alignment environment and inputs"
  require_file "${AISHELL_NER_ENTITY_MANIFEST}" "AISHELL-NER entity-only manifest"
  require_file "${AISHELL_NER_TARGET_CATALOG}" "AISHELL-NER target catalog"
  if is_true "${INSTALL_DEPS}"; then
    "${PYTHON_BIN}" -m pip install -e '.[qwen,config]'
  fi
  "${PYTHON_BIN}" -c 'import importlib.metadata as m; assert m.version("qwen-asr") == "0.0.6", m.version("qwen-asr")'
  "${PYTHON_BIN}" -c 'from qwen_asr import Qwen3ForcedAligner; print(Qwen3ForcedAligner.__name__)'
  if is_true "${REQUIRE_CUDA}"; then
    "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for the aligner"; print(torch.cuda.get_device_name(0))'
  fi
  "${PYTHON_BIN}" scripts/align_hotword_manifest.py --help >/dev/null
  "${PYTHON_BIN}" scripts/validate_aligned_manifest.py --help >/dev/null
}

stage1() {
  echo "[stage1] align transcripts and resolve catalog hotword spans"
  require_file "${AISHELL_NER_ENTITY_MANIFEST}" "AISHELL-NER entity-only manifest"
  require_file "${AISHELL_NER_TARGET_CATALOG}" "AISHELL-NER target catalog"
  local -a optional_args=()
  if is_true "${RESUME_ALIGNMENT}"; then
    optional_args+=(--resume)
  elif is_true "${OVERWRITE_ALIGNMENT}"; then
    optional_args+=(--overwrite)
  fi
  if [[ -n "${ALIGNER_ATTN_IMPLEMENTATION}" ]]; then
    optional_args+=(--attn-implementation "${ALIGNER_ATTN_IMPLEMENTATION}")
  fi
  "${PYTHON_BIN}" scripts/align_hotword_manifest.py \
    --manifest "${AISHELL_NER_ENTITY_MANIFEST}" \
    --catalog "${AISHELL_NER_TARGET_CATALOG}" \
    --output "${AISHELL_NER_ALIGNED_MANIFEST}" \
    --report "${ALIGNMENT_REPORT}" \
    --trace-output "${ALIGNMENT_TRACE}" \
    --model "${ALIGNER_MODEL}" \
    --device "${ALIGNER_DEVICE}" \
    --dtype "${ALIGNER_DTYPE}" \
    --batch-size "${ALIGNER_BATCH_SIZE}" \
    --language "${ALIGNER_LANGUAGE}" \
    --occurrence-policy "${OCCURRENCE_POLICY}" \
    --on-error "${ON_ALIGNMENT_ERROR}" \
    "${optional_args[@]}"
  require_file "${AISHELL_NER_ALIGNED_MANIFEST}" "aligned manifest"
}

stage2() {
  echo "[stage2] validate aligned manifest for boundary generation"
  require_file "${AISHELL_NER_ALIGNED_MANIFEST}" "aligned manifest"
  local -a optional_args=()
  if is_true "${ALLOW_INCOMPLETE}"; then
    optional_args+=(--allow-incomplete)
  fi
  "${PYTHON_BIN}" scripts/validate_aligned_manifest.py \
    --source-manifest "${AISHELL_NER_ENTITY_MANIFEST}" \
    --aligned-manifest "${AISHELL_NER_ALIGNED_MANIFEST}" \
    --catalog "${AISHELL_NER_TARGET_CATALOG}" \
    --report "${ALIGNMENT_VALIDATION_REPORT}" \
    "${optional_args[@]}"
  echo "[done] pass AISHELL_NER_ALIGNED_MANIFEST=${AISHELL_NER_ALIGNED_MANIFEST} to: bash run.sh stage2"
}

run_stage() {
  case "$1" in
    0|stage0) stage0 ;;
    1|stage1) stage1 ;;
    2|stage2) stage2 ;;
    *) echo "unknown stage: $1" >&2; usage; exit 2 ;;
  esac
}

main() {
  if [[ "$#" -eq 0 ]]; then
    usage
    exit 2
  fi
  echo "[run] aligner_model=${ALIGNER_MODEL}"
  echo "[run] input=${AISHELL_NER_ENTITY_MANIFEST}"
  echo "[run] output=${AISHELL_NER_ALIGNED_MANIFEST}"
  if [[ "$1" == "all" ]]; then
    if [[ "$#" -ne 1 ]]; then
      echo "'all' cannot be combined with explicit stages" >&2
      exit 2
    fi
    run_stage stage0
    run_stage stage1
    run_stage stage2
  else
    local stage
    for stage in "$@"; do
      run_stage "${stage}"
    done
  fi
  echo "[done] requested aligner stages completed; log=${RUN_LOG}"
}

main "$@"
