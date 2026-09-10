#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/glclap-multilingual}"
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-384}"
COMPUTE_MATCHED="${COMPUTE_MATCHED:-1}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-0}"
CHUNK_SIZE_SEC="${CHUNK_SIZE_SEC:-2.0}"

AISHELL1_TRAIN_MANIFEST="${AISHELL1_TRAIN_MANIFEST:-${DATA_ROOT}/aishell1/train.jsonl}"
AISHELL2_TRANSCRIPT="${AISHELL2_TRANSCRIPT:-${DATA_ROOT}/aishell2/transcript.txt}"
AISHELL2_WAV_ROOT="${AISHELL2_WAV_ROOT:-${DATA_ROOT}/aishell2/wav}"
MAGICDATA_TRAIN_MANIFEST="${MAGICDATA_TRAIN_MANIFEST:-${DATA_ROOT}/magicdata/train.jsonl}"
HKUST_TRAIN_MANIFEST="${HKUST_TRAIN_MANIFEST:-${DATA_ROOT}/hkust/train.jsonl}"
LIBRI_TRAIN_CLEAN_100="${LIBRI_TRAIN_CLEAN_100:-${DATA_ROOT}/librispeech/train-clean-100.jsonl}"
LIBRI_TRAIN_CLEAN_360="${LIBRI_TRAIN_CLEAN_360:-${DATA_ROOT}/librispeech/train-clean-360.jsonl}"
LIBRI_TRAIN_OTHER_500="${LIBRI_TRAIN_OTHER_500:-${DATA_ROOT}/librispeech/train-other-500.jsonl}"
LIBRI_DEV_CLEAN="${LIBRI_DEV_CLEAN:-${DATA_ROOT}/librispeech/dev-clean.jsonl}"
LIBRI_DEV_OTHER="${LIBRI_DEV_OTHER:-${DATA_ROOT}/librispeech/dev-other.jsonl}"
LIBRI_TEST_CLEAN="${LIBRI_TEST_CLEAN:-${DATA_ROOT}/librispeech/test-clean.jsonl}"
LIBRI_TEST_OTHER="${LIBRI_TEST_OTHER:-${DATA_ROOT}/librispeech/test-other.jsonl}"

AISHELL_NER_DEV_MANIFEST="${AISHELL_NER_DEV_MANIFEST:-${DATA_ROOT}/aishell-ner/dev_entities.jsonl}"
AISHELL_NER_TEST_MANIFEST="${AISHELL_NER_TEST_MANIFEST:-${DATA_ROOT}/aishell-ner/test_entities.jsonl}"
AISHELL_NER_TARGET_CATALOG="${AISHELL_NER_TARGET_CATALOG:-${DATA_ROOT}/aishell-ner/test_targets.jsonl}"
AISHELL_NER_TIMING_MANIFEST="${AISHELL_NER_TIMING_MANIFEST:-}"
LIBRI_TEST_CLEAN_TIMING_MANIFEST="${LIBRI_TEST_CLEAN_TIMING_MANIFEST:-}"
LIBRI_TEST_OTHER_TIMING_MANIFEST="${LIBRI_TEST_OTHER_TIMING_MANIFEST:-}"
STOP1_TIMING_MANIFEST="${STOP1_TIMING_MANIFEST:-}"
STOP2_TIMING_MANIFEST="${STOP2_TIMING_MANIFEST:-}"
HKUST_WORD_FREQ="${HKUST_WORD_FREQ:-${DATA_ROOT}/hkust/word_freq.txt}"
MAGICDATA_WORD_FREQ="${MAGICDATA_WORD_FREQ:-${DATA_ROOT}/magicdata/word_freq.txt}"

STOP1_ROOT="${STOP1_ROOT:-}"
STOP2_ROOT="${STOP2_ROOT:-}"
STOP_TRANSCRIPT_MANIFEST="${STOP_TRANSCRIPT_MANIFEST:-}"

DERIVED="${OUTPUT_ROOT}/data"
TRAIN_ROOT="${OUTPUT_ROOT}/train"
INDEX_ROOT="${OUTPUT_ROOT}/index"
RESULT_ROOT="${OUTPUT_ROOT}/results"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${OUTPUT_ROOT}/feature-cache}"
AISHELL2_MANIFEST="${DERIVED}/aishell2.train.jsonl"
TRAIN_MIX="${DERIVED}/train.zh-en.jsonl"
ZH_NEGATIVE_POOL="${DERIVED}/zh.train-neg.10k.jsonl"
EN_NEGATIVE_POOL="${DERIVED}/en.train-neg.10k.jsonl"
ZH_EVAL_CATALOG="${DERIVED}/zh.eval.10k.jsonl"
EN_EVAL_CATALOG="${DERIVED}/en.eval.10k.jsonl"
MIXED_EVAL_CATALOG="${DERIVED}/zh-en.eval.20k.jsonl"

mkdir -p "${DERIVED}" "${TRAIN_ROOT}" "${INDEX_ROOT}" "${RESULT_ROOT}"

libri_train_args() {
  printf '%s\n' \
    "--train-manifest" "${LIBRI_TRAIN_CLEAN_100}" \
    "--train-manifest" "${LIBRI_TRAIN_CLEAN_360}" \
    "--train-manifest" "${LIBRI_TRAIN_OTHER_500}"
}

count_jsonl() {
  "${PYTHON_BIN}" -c 'import sys; print(sum(bool(line.strip()) for line in open(sys.argv[1], encoding="utf-8")))' "$1"
}

stage0() {
  echo "[stage0] branch and input preflight"
  test "$(git branch --show-current)" = "feat/glclap-training-streamingdecode"
  local path
  for path in \
    "${AISHELL1_TRAIN_MANIFEST}" "${AISHELL2_TRANSCRIPT}" "${AISHELL2_WAV_ROOT}" \
    "${MAGICDATA_TRAIN_MANIFEST}" "${HKUST_TRAIN_MANIFEST}" \
    "${LIBRI_TRAIN_CLEAN_100}" "${LIBRI_TRAIN_CLEAN_360}" "${LIBRI_TRAIN_OTHER_500}" \
    "${LIBRI_DEV_CLEAN}" "${LIBRI_DEV_OTHER}" "${LIBRI_TEST_CLEAN}" "${LIBRI_TEST_OTHER}" \
    "${AISHELL_NER_DEV_MANIFEST}" "${AISHELL_NER_TEST_MANIFEST}" \
    "${AISHELL_NER_TARGET_CATALOG}" "${HKUST_WORD_FREQ}" "${MAGICDATA_WORD_FREQ}"; do
    test -e "${path}" || { echo "missing input: ${path}" >&2; exit 2; }
  done
}

stage1() {
  echo "[stage1] prepare AISHELL-2 and normalized training mix"
  "${PYTHON_BIN}" scripts/prepare_aishell2_manifest.py \
    --transcript "${AISHELL2_TRANSCRIPT}" --wav-root "${AISHELL2_WAV_ROOT}" \
    --output "${AISHELL2_MANIFEST}" --report "${AISHELL2_MANIFEST}.report.json"
  "${PYTHON_BIN}" scripts/build_glclap_training_mix.py --check-audio \
    --input "aishell1:zh:train=${AISHELL1_TRAIN_MANIFEST}" \
    --input "aishell2:zh:train=${AISHELL2_MANIFEST}" \
    --input "magicdata:zh:train=${MAGICDATA_TRAIN_MANIFEST}" \
    --input "hkust:zh:train=${HKUST_TRAIN_MANIFEST}" \
    --input "librispeech:en:train-clean-100=${LIBRI_TRAIN_CLEAN_100}" \
    --input "librispeech:en:train-clean-360=${LIBRI_TRAIN_CLEAN_360}" \
    --input "librispeech:en:train-other-500=${LIBRI_TRAIN_OTHER_500}" \
    --output "${TRAIN_MIX}" --report "${TRAIN_MIX}.report.json"
}

stage2() {
  echo "[stage2] build language-specific training negative pools"
  "${PYTHON_BIN}" scripts/build_glclap_training_pool.py \
    --word-freq "hkust=${HKUST_WORD_FREQ}" --word-freq "magicdata=${MAGICDATA_WORD_FREQ}" \
    --output "${ZH_NEGATIVE_POOL}" --report "${ZH_NEGATIVE_POOL}.report.json" --seed "${SEED}"
  mapfile -t train_args < <(libri_train_args)
  "${PYTHON_BIN}" scripts/build_english_glclap_pool.py "${train_args[@]}" \
    --output "${EN_NEGATIVE_POOL}" --report "${EN_NEGATIVE_POOL}.report.json" --seed "${SEED}"
}

prepare_libri_split() {
  local name="$1" input="$2"
  mapfile -t train_args < <(libri_train_args)
  "${PYTHON_BIN}" scripts/prepare_librispeech_hotword_eval.py "${train_args[@]}" \
    --eval-manifest "${input}" --split "${name}" \
    --manifest-output "${DERIVED}/librispeech.${name}.eval.jsonl" \
    --target-catalog-output "${DERIVED}/librispeech.${name}.targets.jsonl" \
    --report "${DERIVED}/librispeech.${name}.report.json" --seed "${SEED}" \
    --target-vocabulary "${EN_NEGATIVE_POOL}" --max-target-vocabulary 8000
}

prepare_stop() {
  local dataset="$1" root="$2"
  test -n "${root}" || return 0
  local transcript_args=()
  test -z "${STOP_TRANSCRIPT_MANIFEST}" || transcript_args=(--transcript-manifest "${STOP_TRANSCRIPT_MANIFEST}")
  "${PYTHON_BIN}" scripts/prepare_stop_hotword_eval.py \
    --dataset "${dataset}" --wav-scp "${root}/wav.scp" \
    --hotlists "${root}/hotlists" --hotlists-uniq "${root}/hotlists.uniq" \
    --manifest-output "${DERIVED}/${dataset}.eval.jsonl" \
    --target-catalog-output "${DERIVED}/${dataset}.targets.jsonl" \
    --report "${DERIVED}/${dataset}.report.json" "${transcript_args[@]}"
}

stage3() {
  echo "[stage3] freeze LibriSpeech synthetic and STOP labels"
  prepare_libri_split dev-clean "${LIBRI_DEV_CLEAN}"
  prepare_libri_split dev-other "${LIBRI_DEV_OTHER}"
  prepare_libri_split test-clean "${LIBRI_TEST_CLEAN}"
  prepare_libri_split test-other "${LIBRI_TEST_OTHER}"
  prepare_stop stop1 "${STOP1_ROOT}"
  prepare_stop stop2 "${STOP2_ROOT}"
}

stage4() {
  echo "[stage4] build zh 10k, en 10k, and bilingual 20k catalogs"
  "${PYTHON_BIN}" scripts/build_glclap_evaluation_catalog.py \
    --word-freq "hkust=${HKUST_WORD_FREQ}" --word-freq "magicdata=${MAGICDATA_WORD_FREQ}" \
    --target-catalog "${AISHELL_NER_TARGET_CATALOG}" --eval-manifest "${AISHELL_NER_TEST_MANIFEST}" \
    --output "${ZH_EVAL_CATALOG}" --report "${ZH_EVAL_CATALOG}.report.json" --seed "${SEED}"
  local target_args=() manifest_args=() name
  for name in dev-clean dev-other test-clean test-other; do
    target_args+=(--input "${DERIVED}/librispeech.${name}.targets.jsonl")
    manifest_args+=(--input "${DERIVED}/librispeech.${name}.eval.jsonl")
  done
  for name in stop1 stop2; do
    if test -f "${DERIVED}/${name}.targets.jsonl"; then
      target_args+=(--input "${DERIVED}/${name}.targets.jsonl")
      manifest_args+=(--input "${DERIVED}/${name}.eval.jsonl")
    fi
  done
  "${PYTHON_BIN}" scripts/merge_glclap_catalogs.py "${target_args[@]}" \
    --output "${DERIVED}/en.all.targets.jsonl" --report "${DERIVED}/en.all.targets.report.json" \
    --version en-all-targets-v1
  "${PYTHON_BIN}" scripts/merge_glclap_manifests.py "${manifest_args[@]}" \
    --output "${DERIVED}/en.all.eval.jsonl" --report "${DERIVED}/en.all.eval.report.json"
  "${PYTHON_BIN}" scripts/build_glclap_catalog_from_pool.py \
    --target-catalog "${DERIVED}/en.all.targets.jsonl" --distractor-pool "${EN_NEGATIVE_POOL}" \
    --eval-manifest "${DERIVED}/en.all.eval.jsonl" --language en --size 10000 \
    --version en-eval-10k-v1 --output "${EN_EVAL_CATALOG}" --report "${EN_EVAL_CATALOG}.report.json" --seed "${SEED}"
  "${PYTHON_BIN}" scripts/merge_glclap_catalogs.py \
    --input "${ZH_EVAL_CATALOG}" --input "${EN_EVAL_CATALOG}" \
    --output "${MIXED_EVAL_CATALOG}" --report "${MIXED_EVAL_CATALOG}.report.json" \
    --version zh-en-eval-20k-v1
}

train_variant() {
  local variant="$1" config="$2" output="${TRAIN_ROOT}/${variant}"
  local epoch_samples="${SAMPLES_PER_EPOCH}"
  if test "${COMPUTE_MATCHED}" = "1" && test "${epoch_samples}" = "0"; then
    epoch_samples="$(count_jsonl "${AISHELL1_TRAIN_MANIFEST}")"
  fi
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
    scripts/train_glclap_retriever.py --config "${config}" --manifest "${TRAIN_MIX}" \
    --dev-manifest "aishell-ner=${AISHELL_NER_DEV_MANIFEST}" \
    --dev-manifest "librispeech-dev-clean=${DERIVED}/librispeech.dev-clean.eval.jsonl" \
    --dev-manifest "librispeech-dev-other=${DERIVED}/librispeech.dev-other.eval.jsonl" \
    --negative-catalog "zh=${ZH_NEGATIVE_POOL}" --negative-catalog "en=${EN_NEGATIVE_POOL}" \
    --feature-cache-dir "${FEATURE_CACHE_DIR}" --output-dir "${output}" \
    --override "training.samples_per_epoch=${epoch_samples}" \
    --override "training.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --override "training.seed=${SEED}" --override "evaluation.seed=${SEED}"
}

variant_config() {
  case "$1" in
    global-only) printf '%s\n' configs/glclap/multilingual_global_only.yaml ;;
    frozen) printf '%s\n' configs/glclap/multilingual_frozen.yaml ;;
    *) echo "unknown training variant: $1" >&2; return 2 ;;
  esac
}

stage5() {
  echo "[stage5] train compute-matched multilingual variants"
  train_variant global-only configs/glclap/multilingual_global_only.yaml
  train_variant frozen configs/glclap/multilingual_frozen.yaml
}

stage6() {
  echo "[stage6] build monolingual and bilingual indexes"
  local variant catalog label
  for variant in global-only frozen; do
    local config
    config="$(variant_config "${variant}")"
    for label in zh en mixed; do
      case "${label}" in
        zh) catalog="${ZH_EVAL_CATALOG}" ;;
        en) catalog="${EN_EVAL_CATALOG}" ;;
        mixed) catalog="${MIXED_EVAL_CATALOG}" ;;
      esac
      "${PYTHON_BIN}" scripts/build_glclap_index.py \
        --config "${config}" \
        --checkpoint "${TRAIN_ROOT}/${variant}/best.pt" --catalog "${catalog}" \
        --output "${INDEX_ROOT}/${variant}.${label}.npz"
    done
  done
}

eval_dataset() {
  local name="$1" language="$2" manifest="$3" variant="$4" timing_manifest="${5:-}"
  local config
  config="$(variant_config "${variant}")"
  local mode mono mixed
  for mode in offline streaming; do
    mono="${RESULT_ROOT}/${variant}.${name}.${mode}.mono.jsonl"
    mixed="${RESULT_ROOT}/${variant}.${name}.${mode}.mixed.jsonl"
    "${PYTHON_BIN}" scripts/decode_streaming_retrieval.py --config "${config}" \
      --checkpoint "${TRAIN_ROOT}/${variant}/best.pt" --manifest "${manifest}" \
      --index "${INDEX_ROOT}/${variant}.${language}.npz" --output "${mono}" \
      --mode "${mode}" --require-target-coverage --override "streaming.chunk_size_sec=${CHUNK_SIZE_SEC}"
    "${PYTHON_BIN}" scripts/decode_streaming_retrieval.py --config "${config}" \
      --checkpoint "${TRAIN_ROOT}/${variant}/best.pt" --manifest "${manifest}" \
      --index "${INDEX_ROOT}/${variant}.mixed.npz" --output "${mixed}" \
      --mode "${mode}" --require-target-coverage --override "streaming.chunk_size_sec=${CHUNK_SIZE_SEC}"
    local baseline_args=()
    if test "${variant}" = frozen; then
      baseline_args=(--baseline "${RESULT_ROOT}/global-only.${name}.${mode}.mixed.jsonl")
    fi
    local online_args=()
    if test "${mode}" = streaming && test -n "${timing_manifest}"; then
      online_args=(--online --timing-manifest "${timing_manifest}")
    fi
    "${PYTHON_BIN}" scripts/eval_hotword_retrieval.py --input "${mixed}" \
      --monolingual-reference "${mono}" "${baseline_args[@]}" \
      "${online_args[@]}" \
      --output "${mixed}.metrics.json" --seed "${SEED}"
  done
}

stage7() {
  echo "[stage7] evaluate every dataset against mono and mixed catalogs"
  local variant name
  for variant in global-only frozen; do
    eval_dataset aishell-ner zh "${AISHELL_NER_TEST_MANIFEST}" "${variant}" "${AISHELL_NER_TIMING_MANIFEST}"
    eval_dataset librispeech-test-clean en "${DERIVED}/librispeech.test-clean.eval.jsonl" "${variant}" "${LIBRI_TEST_CLEAN_TIMING_MANIFEST}"
    eval_dataset librispeech-test-other en "${DERIVED}/librispeech.test-other.eval.jsonl" "${variant}" "${LIBRI_TEST_OTHER_TIMING_MANIFEST}"
    test ! -f "${DERIVED}/stop1.eval.jsonl" || eval_dataset stop1 en "${DERIVED}/stop1.eval.jsonl" "${variant}" "${STOP1_TIMING_MANIFEST}"
    test ! -f "${DERIVED}/stop2.eval.jsonl" || eval_dataset stop2 en "${DERIVED}/stop2.eval.jsonl" "${variant}" "${STOP2_TIMING_MANIFEST}"
  done
  "${PYTHON_BIN}" scripts/summarize_multilingual_retrieval.py \
    --results-dir "${RESULT_ROOT}" --output "${OUTPUT_ROOT}/multilingual-summary.json"
}

stage8() {
  echo "[stage8] dependency-free tests"
  "${PYTHON_BIN}" -m unittest discover -s tests -v
}

run_stage() {
  case "$1" in
    0|stage0) stage0 ;; 1|stage1) stage1 ;; 2|stage2) stage2 ;;
    3|stage3) stage3 ;; 4|stage4) stage4 ;; 5|stage5) stage5 ;;
    6|stage6) stage6 ;; 7|stage7) stage7 ;; 8|stage8) stage8 ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

if test "$#" -eq 0; then
  echo "usage: $0 all|stage0 ... stage8" >&2
  exit 2
fi
if test "$1" = all; then
  for stage in {0..8}; do run_stage "stage${stage}"; done
else
  for stage in "$@"; do run_stage "${stage}"; done
fi
