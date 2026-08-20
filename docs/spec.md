# Qwen3-ASR Contextual Streaming Specification

## 1. Objective

The primary system is Qwen3-ASR accumulated-audio streaming with a persistent
contextual path. The research hypothesis is that cross-chunk hotword state
should determine the mutable transcript frontier instead of a fixed token count.

The existing CTC modules remain source-compatible and are not the primary text
decoder. They may be reused for auxiliary probes.

## 2. Runtime flow

1. Buffer arbitrary 16 kHz PCM until the configured effective chunk is ready.
2. Append the chunk to accumulated audio.
3. Run the optional acoustic probe on new audio plus bounded left overlap.
4. Continue stateful pronunciation-trie paths and update candidate lifecycle.
5. Calculate the text prefix using committed tokens and active candidate anchors.
6. Build a small dynamic context and request-local sparse logit processor.
7. Re-decode accumulated audio through the pinned Qwen vLLM adapter.
8. Observe the refreshed transcript, update text candidates, and advance only a safe stable prefix.
9. Emit partial/stable text and a complete trace record.

## 3. Public interfaces

### `HotwordEntry`

Fields: `id`, `text`, `aliases`, `language`, `weight`, and JSON-compatible
`metadata`. Catalog JSONL records also contain `catalog_version`.

### `ContextualStreamingSession`

- `step(pcm16k) -> StreamingResult`
- `finish() -> StreamingResult`
- `reset() -> None`
- `update_hotwords(enabled_ids=None, overlay_entries=()) -> None`

Updates are queued until the next processed chunk. Removed entries expire
immediately at that boundary. Overlay entries match only future input; no audio
or transcript history is rescanned.

### `StreamingResult`

Contains partial text, stable text, newly committed text, active/confirmed
candidate snapshots, rollback start token, chunk ID, processing time, final
flag, and debug instrumentation.

### Probe and retriever

`AcousticProbe.process_pcm` emits `ProbeChunk(log_probs [T,P], frame_offset,
blank_id, symbols)`. `CTCPosteriorTrieRetriever.consume` maintains at most 256
paths and returns active/completed evidence. `TranscriptProbe` is the no-training
M1 implementation.

## 4. Commitment contract

Let `N` be the previous output length, `C` the committed prefix length,
`K_base=5`, `K_max=32`, and `A` the valid active anchors. After the first two
unfixed chunks:

```text
baseline = max(C, min(1, N), N - K_base)
rollback_start = max(C, N - K_max, min(baseline, min(A)))  # when A is non-empty
rollback_start = baseline                                  # otherwise
```

The `min(1,N)` term matches the pinned official Qwen behavior for short
hypotheses. A backend rewriting an already committed prefix is an invariant
failure, not a recoverable revision.

## 5. Candidate and bias policy

- lifecycle: inactive → active → confirmed/rejected/expired
- default enter/exit thresholds: 0.55/0.35
- active TTL: 3 chunks; confirmed prompt TTL: 2 chunks
- maximum live retrieval candidates: 20; prompt candidates: 5
- token bias default/cap: 2.0/4.0, scaled by acoustic confidence and entry weight
- token paths cover raw, normalized, leading-space, newline, and alias forms

Thresholds are selected on dev data by maximum hotword F1 subject to at most
0.5 percentage-point absolute UWER degradation, then frozen for test.

## 6. Compatibility and limits

- Linux CUDA integration is pinned to qwen-asr 0.0.6 and vLLM 0.14.0.
- Qwen/vLLM/PyTorch imports are optional for the core package.
- Streaming remains single-session per official decoder state; independent
  sessions may run in separate workers for experiment throughput.
- Online timestamps, batching, production serving, full Qwen SFT, and GRPO are
  outside the first-paper implementation.
