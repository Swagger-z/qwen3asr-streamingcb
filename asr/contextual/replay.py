"""Single-stream replay clocks; no torch dependency and no background workers."""

from __future__ import annotations

import math
import time
from typing import Callable


class ReplayClock:
    """Timestamp refreshes against scheduled PCM arrivals on a [0, duration] clock.

    ``fast`` measures service durations and simulates a FIFO. ``realtime`` uses
    an actual monotonic clock. Input audio is preloaded; no network/capture time
    is claimed. Inject ``now``/``sleep`` for deterministic, non-sleeping tests.
    """

    def __init__(self, total_samples: int, feed_samples: int, *, mode: str = "fast",
                 sample_rate: int = 16000, now: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if mode not in {"fast", "realtime"}:
            raise ValueError("replay mode must be fast or realtime")
        if total_samples <= 0 or feed_samples <= 0 or sample_rate <= 0:
            raise ValueError("replay requires nonempty audio and positive feed/sample rate")
        self.total_samples, self.feed_samples = int(total_samples), int(feed_samples)
        self.sample_rate, self.mode = int(sample_rate), mode
        self.now, self.sleep = now, sleep
        self.origin = now()
        self.previous_finish = 0.0

    def reset(self) -> None:
        """Start a new utterance on the same delivery schedule."""
        self.origin = self.now()
        self.previous_finish = 0.0

    def wait_for_feed(self, received_samples: int) -> None:
        """Deliver a feed at its absolute scheduled time, never sleep after work."""
        if self.mode == "realtime":
            deadline = self.origin + received_samples / self.sample_rate
            while (remaining := deadline - self.now()) > 0:
                self.sleep(min(remaining, 0.1))

    def start(self, sample_count: int) -> tuple[float, float, float]:
        """Return service timer, audio cutoff, and scheduled task-ready time."""
        if not 0 < sample_count <= self.total_samples:
            raise ValueError("refresh cutoff outside input audio")
        ready_samples = min(self.total_samples,
                            math.ceil(sample_count / self.feed_samples) * self.feed_samples)
        return self.now(), sample_count / self.sample_rate, ready_samples / self.sample_rate

    def finish(self, started: tuple[float, float, float]) -> dict[str, float | str]:
        """Record actual service time and FIFO/observed result availability."""
        before, cutoff, ready = started
        after = self.now()
        duration = max(0.0, after - before)
        start = max(ready, self.previous_finish) if self.mode == "fast" else before - self.origin
        finish = start + duration if self.mode == "fast" else after - self.origin
        if start + 1e-7 < ready:
            raise ValueError("refresh ran before its audio arrived")
        self.previous_finish = finish
        return {
            "clock": "simulated_fifo" if self.mode == "fast" else "monotonic_realtime",
            "audio_cutoff_sec": cutoff,
            "ready_sec": ready,
            "start_sec": start,
            "finish_sec": finish,
            "processing_sec": duration,
            "feed_wait_sec": ready - cutoff,
            "queue_wait_sec": max(0.0, start - ready),
        }
