# Stateful Contextual Streaming Design

## State ownership

- Backend: official tokenizer/model handles and accumulated audio decoding only.
- Session: PCM buffer, accumulated audio, previous raw tokens, chunk index, and traces.
- Probe: 200 ms PCM overlap and absolute emitted phoneme frame index.
- Retriever: AC/CTC trie paths, blank/repeat state, scores, start frame, anchor token.
- Candidate manager: lifecycle, confidence hysteresis, TTL, per-session enabled IDs.
- Commit controller: immutable committed token prefix.
- Token bias: pure per-request snapshot; no process-global mutable candidate state.

## Per-chunk sequence

Acoustic evidence is consumed before decode so existing candidates can affect the
current prefix, prompt, and logits. Transcript evidence is consumed after decode;
newly opened text candidates immediately hold commitment and influence the next
refresh. This prevents committing a candidate merely because it was discovered
after the decode call.

## Dynamic catalog updates

A global 1k–100k catalog is built once. Each session filters it by ID. Small
overlays rebuild the affected trie snapshot at the next chunk, reset acoustic
token paths, and retain already managed candidates until normal expiry. Historical
input is never replayed for newly added entries.

## Adaptive lookahead

Disabled in the main experiment. When enabled, the next input threshold grows by
100/200/300 ms only if a holding candidate confidence lies in the configured
uncertainty interval. `chunk_size_ms + lookahead_ms` may not exceed
`max_effective_context_ms`. Confirmed, rejected, very low-confidence, and
high-confidence candidates do not spend extra acquisition latency.

## Trace fields

Every processed chunk records audio end time, raw output/tokens, display and
stable text, decode and next rollback positions, committed delta, candidate
states, prompt hotword IDs/tokens, active retriever state count, revision count,
language, and processing latency. `finish()` writes a final record.

## Failure handling

- committed-prefix rewrite: raise `CommitInvariantError`
- catalog/version mismatch: fail before session start
- missing pinned Qwen/vLLM/PyTorch dependency: explicit import/version error
- unsupported WAV/sample rate: explicit input error
- active path overflow: score-prune to 256 states
- false active candidate: hysteresis, 3-chunk timeout, and 32-token rollback cap
