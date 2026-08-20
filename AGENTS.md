# Project: qwen3-asr-contextual-biasing

## Goal
Build a research-grade streaming LLM-ASR system on Qwen3-ASR whose main
contribution is cross-chunk contextual state controlling stable commitment and
bounded token rollback.

## Current milestones
- M0: pinned official Qwen3-ASR/vLLM baseline, trace instrumentation, and boundary-stress data tooling
- M1: stateful transcript matching plus hotword-aware commit/rollback without training
- M2: frozen AuT sidecar with a trainable phoneme CTC head and stateful posterior-trie retrieval
- M3: confidence-gated sparse token bias, small dynamic prompt, and optional adaptive lookahead

## Scope
- accumulated-audio Qwen3-ASR streaming inference
- per-session hotword catalogs from 1k to 100k entries
- Chinese primary experiments and English replication
- training, decoding, evaluation, tracing, and controlled boundary scripts

## Non-goals for the first paper
- full Qwen SFT or GRPO
- production serving optimization
- online timestamps or forced alignment
- batched streaming sessions
- complex LM rescoring

Offline Qwen3-ForcedAligner may be used only to prepare boundary-controlled datasets.

## Architecture constraints
- keep the external Qwen backend, probe, retriever, candidate lifecycle, token bias, and commit controller decoupled
- never patch installed Qwen or vLLM source in place; use the pinned adapter
- preserve the original CTC package interfaces for compatibility and probe research
- keep all candidate and decoder state session-local
- keep active states, rollback distance, prompt size, and caches explicitly bounded
- use config-driven entrypoints and dataset-agnostic JSONL schemas

## Coding rules
- make small, reviewable commits when the workspace is a Git repository
- do not silently change public interfaces
- add docstrings to public classes/functions
- add basic tests for every core behavior
- use explicit tensor shapes in comments and docstrings
- keep PyTorch, Qwen, and vLLM imports optional outside GPU entrypoints

## Done means
- dependency-free core tests pass with `python -m unittest discover -s tests -v`
- the 100k test passes with `RUN_CONTEXTUAL_STRESS=1`
- pinned Linux CUDA integration tests pass when model weights are available
- README and design docs contain exact reproduction commands
- experiment outputs record resolved config, model/catalog versions, seed, and traces
