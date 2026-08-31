"""Online retrieval evaluation tests with deterministic clocks."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from asr.contextual.glclap import AudioEncoding, HotwordEmbeddingIndex
from asr.contextual.replay import ReplayClock
from asr.data.manifest import manifest_source
from asr.data.timed_entities import prepare_timed_records, validate_timed_records
from asr.eval.online_retrieval_metrics import (
    attach_entity_timing, evaluate_online_records, _boundary_pairs, _cluster_ci,
)
from scripts.decode_streaming_retrieval import _retrieve_waveform

ROOT = Path(__file__).resolve().parents[1]


class FakeTime:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class TimedEncoder:
    def __init__(self, timer, service=0.1):
        self.timer, self.service, self.lengths = timer, service, []

    def encode_pcm(self, waveform):
        self.lengths.append(len(waveform))
        self.timer.advance(self.service)
        return AudioEncoding(np.asarray([[1.0, 0.0]], dtype=np.float32))


def example_record(predictions=None, *, key="u1", entities=None, service=0.1):
    """Create a six-second record with three refreshes."""
    timer = FakeTime()
    clock = ReplayClock(96000, 1600, now=timer.now, sleep=timer.advance)
    batches = []
    predictions = predictions if predictions is not None else [[], ["gold"], ["gold"]]
    for j, ids in enumerate(predictions):
        stamp = clock.start((j + 1) * 32000)
        timer.advance(service)
        batches.append({
            "chunk_id": j, "is_final": j == 2,
            "accumulated_audio_sec": (j + 1) * 2, "timeline": clock.finish(stamp),
            "hits": [{"hotword_id": hid, "rank": rank} for rank, hid in enumerate(ids, 1)],
        })
    entities = entities or [
        {"mention_id": "m1", "hotword_id": "gold", "start_sec": 1.8, "end_sec": 2.2}
    ]
    return {
        "key": key, "utt_id": key, "source_utt_id": key,
        "source": "/audio/u1.wav", "audio": "/audio/u1.wav",
        "audio_sha256": "a" * 64, "duration_sec": 6.0, "timing_schema_version": 2,
        "target_hotword_ids": sorted({e["hotword_id"] for e in entities}), "entities": entities,
        "retrieval_mode": "streaming", "replay_mode": "fast", "retrieval_top_k": 50,
        "feed_step_ms": 100, "chunk_size_sec": 2, "batches": batches, "final_batch": batches[-1],
    }


class OnlineMetricTests(unittest.TestCase):
    def evaluate(self, records, **kwargs):
        return evaluate_online_records(records, ks=(1, 5, 50), bootstrap_samples=40, **kwargs)

    def test_latency_decomposition_and_deadline_equality(self):
        summary, details, refreshes = self.evaluate([example_record()], deadlines_ms=(1000, 1900, 2000))
        metrics = summary["by_k"]["1"]
        self.assertEqual(metrics["first_complete_refresh_recall"], 1)
        self.assertAlmostEqual(metrics["latency_ms_mean"], 1900)
        self.assertAlmostEqual(metrics["audio_wait_ms_mean"], 1800)
        self.assertAlmostEqual(metrics["processing_ms_mean"], 100)
        self.assertEqual(metrics["deadline_recall"], {"1000": 0.0, "1900": 1.0, "2000": 1.0})
        self.assertIsNone(refreshes[0]["recall_at_1"])
        row = details[0]
        self.assertAlmostEqual(row["latency_ms"], sum(row[name] for name in
                               ("audio_wait_ms", "feed_wait_ms", "queue_wait_ms", "processing_ms")))

    def test_early_candidates_and_miss_denominator(self):
        miss = example_record([["gold"], [], []], key="miss")
        success = example_record(key="success")
        summary, rows, _ = self.evaluate([miss, success])
        metrics = summary["by_k"]["1"]
        self.assertEqual(metrics["target_count"], 2)
        self.assertEqual(metrics["miss_rate"], 0.5)
        self.assertEqual(metrics["deadline_recall"]["2000"], 0.5)
        self.assertEqual(metrics["early_during_word_rate"], 0.5)
        self.assertIsNone(rows[0]["latency_ms"])
        all_miss, _, _ = self.evaluate([miss])
        self.assertIsNone(all_miss["by_k"]["1"]["latency_ms_p95"])

    def test_rank_dropout_and_repeated_availability(self):
        record = example_record([["other", "gold"], ["other", "gold"], []])
        summary, _, _ = self.evaluate([record])
        self.assertEqual(summary["by_k"]["1"]["miss_rate"], 1)
        self.assertEqual(summary["by_k"]["5"]["dropout_rate"], 1)
        record = example_record(entities=[
            {"mention_id": "first", "hotword_id": "gold", "start_sec": 1.8, "end_sec": 2.2},
            {"mention_id": "repeat", "hotword_id": "gold", "start_sec": 4.5, "end_sec": 4.7},
        ])
        summary, rows, _ = self.evaluate([record])
        self.assertEqual(summary["by_k"]["1"]["target_count"], 1)
        self.assertEqual(summary["by_k"]["1"]["repeat_candidate_availability"], 1)
        self.assertTrue(all(row["latency_ms"] is None for row in rows if not row["primary"]))

    def test_pre_onset_and_non_target_future_entities(self):
        record = example_record([["gold"], [], []], entities=[
            {"mention_id": "late", "hotword_id": "gold", "start_sec": 3, "end_sec": 3.5},
        ])
        summary, _, refresh = self.evaluate([record])
        self.assertEqual(summary["by_k"]["1"]["early_before_onset_rate"], 1)
        self.assertEqual(refresh[0]["completed_primary_count"], 0)

    def test_invalid_inputs_are_rejected(self):
        for change in (
            lambda r: r.pop("timing_schema_version"),
            lambda r: r["batches"][0].pop("timeline"),
            lambda r: r["batches"].pop(0),
            lambda r: r.update(source="different.wav"),
            lambda r: r.update(retrieval_top_k=10),
            lambda r: r["batches"][1]["timeline"].update(processing_sec=-1),
        ):
            record = example_record()
            change(record)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                self.evaluate([record])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.evaluate([example_record(), example_record()])

    def test_verified_timing_join_and_cluster_bootstrap(self):
        timed = example_record()
        result = copy.deepcopy(timed)
        result.pop("entities")
        joined = attach_entity_timing([result], [timed])
        self.assertEqual(joined[0]["entities"], timed["entities"])
        result.pop("audio_sha256")
        with self.assertRaisesRegex(ValueError, "verify"):
            attach_entity_timing([result], [timed])
        groups = {"u1": [1, 1, 1], "u2": [1]}
        ci = _cluster_ci(groups, 40, 42)
        self.assertEqual(ci["source_utterance_count"], 2)
        self.assertEqual(ci["paired_focus_count"], 4)
        self.assertEqual(ci["ci95_low"], 1)
        self.assertEqual(ci, _cluster_ci(groups, 40, 42))

    def test_paired_boundary_penalty_clusters_all_mentions_of_a_source(self):
        rows = []
        for source, mention, center_hit in (("u1", "m1", True), ("u1", "m2", True), ("u2", "m1", False)):
            for label in ("center", "cross-25", "cross-50", "cross-75"):
                hit = center_hit if label == "center" else not center_hit
                rows.append({"source_utt_id": source, "mention_id": mention, "primary": True,
                             "boundary_group": label, "first_complete_hit": hit,
                             "deadline_hits": {"2000": hit}})
        rows.append({**rows[0], "mention_id": "unpaired"})
        result = _boundary_pairs(rows, (2000,), 200, 42)
        self.assertEqual(result["unpaired_focus_count"], 1)
        first = result["first_complete_refresh"]
        self.assertEqual(first["source_utterance_count"], 2)
        self.assertEqual(first["paired_focus_count"], 3)
        self.assertAlmostEqual(first["estimate"], 1 / 3)
        self.assertEqual(first, result["deadline_2000ms"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _boundary_pairs(rows + rows[:1], (2000,), 10, 42)
        missing = _boundary_pairs(rows[-1:], (2000,), 10, 42)
        self.assertIsNone(missing["first_complete_refresh"]["estimate"])

    def test_timing_join_cannot_relabel_targets_or_boundary_identity(self):
        reference = example_record()
        for update in ({"target_hotword_ids": ["wrong"]}, {"source_utt_id": "wrong"},
                       {"boundary_group": "cross-25"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                attach_entity_timing([{**reference, **update}], [reference])

    def test_old_records_still_have_sentence_final_metrics(self):
        from asr.eval.glclap_metrics import evaluate_glclap_records
        record = {"utt_id": "legacy", "target_hotword_ids": ["gold"],
                  "final_hits": [{"hotword_id": "gold"}]}
        self.assertEqual(evaluate_glclap_records([record])["recall_at_1"], 1)
        with self.assertRaisesRegex(ValueError, "timing_schema_version"):
            self.evaluate([record])


class ReplayTests(unittest.TestCase):
    def replay(self, mode, *, duration=5.3, chunk=2, feed=0.1, service=0.1):
        timer = FakeTime()
        encoder = TimedEncoder(timer, service)
        index = HotwordEmbeddingIndex(np.eye(2, dtype=np.float32), ["gold", "other"], ["G", "O"])
        count, feed_samples = round(duration * 16000), round(feed * 16000)
        clock = ReplayClock(count, feed_samples, mode=mode, now=timer.now, sleep=timer.advance)
        batches = _retrieve_waveform(
            encoder, index, [0.0] * count, mode="streaming", chunk_size_sec=chunk,
            feed_samples=feed_samples, top_k=2, replay_clock=clock,
        )
        return encoder, batches

    def test_fast_and_realtime_match_with_slow_worker_and_tail(self):
        fast_encoder, fast = self.replay("fast", service=2.5)
        real_encoder, real = self.replay("realtime", service=2.5)
        self.assertEqual(fast_encoder.lengths, [32000, 64000, 84800])
        self.assertEqual(fast_encoder.lengths, real_encoder.lengths)
        self.assertEqual([b.hits for b in fast], [b.hits for b in real])
        for simulated, measured in zip(fast, real):
            for key in ("ready_sec", "start_sec", "finish_sec", "processing_sec"):
                self.assertAlmostEqual(simulated.timeline[key], measured.timeline[key])
        self.assertAlmostEqual(real[1].timeline["queue_wait_sec"], 0.5)
        self.assertAlmostEqual(real[-1].timeline["queue_wait_sec"], 1.7)
        self.assertTrue(real[-1].is_final)

    def test_multiple_refreshes_in_one_feed_have_distinct_service_times(self):
        _, batches = self.replay("realtime", feed=5)
        self.assertEqual([b.timeline["ready_sec"] for b in batches], [5, 5, 5.3])
        self.assertAlmostEqual(batches[0].timeline["finish_sec"], 5.1)
        self.assertAlmostEqual(batches[1].timeline["start_sec"], 5.1)

    def test_non_divisible_feed_and_exact_tail(self):
        for mode in ("fast", "realtime"):
            _, batches = self.replay(mode, duration=2, chunk=0.56)
            self.assertEqual([b.accumulated_audio_sec for b in batches], [0.56, 1.12, 1.68, 2.0])
            self.assertEqual([b.timeline["ready_sec"] for b in batches], [0.6, 1.2, 1.7, 2.0])
            encoder, batches = self.replay(mode, duration=4)
            self.assertEqual(encoder.lengths, [32000, 64000])
            self.assertEqual(sum(b.is_final for b in batches), 1)

    def test_session_reset_clears_replay_queue(self):
        from asr.contextual.glclap import AccumulatedAudioRetrievalSession
        timer = FakeTime()
        clock = ReplayClock(32000, 1600, now=timer.now, sleep=timer.advance)
        index = HotwordEmbeddingIndex(np.eye(2, dtype=np.float32), ["gold", "other"], ["G", "O"])
        session = AccumulatedAudioRetrievalSession(TimedEncoder(timer, 3), index, refresh_clock=clock)
        first = session.step([0] * 32000)[0]
        session.finish()
        session.reset()
        second = session.step([0] * 32000)[0]
        self.assertEqual(first.timeline, second.timeline)


class TimedDataTests(unittest.TestCase):
    def fixtures(self, directory):
        audio = directory / "u1.wav"
        with wave.open(str(audio), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\x01\x00" * 96000)
        mentions = [
            {"mention_id": "u1/m1", "hotword_id": "gold", "occurrence_index": 0},
            {"mention_id": "u1?m1", "hotword_id": "gold", "occurrence_index": 1},
        ]
        source = {"key": "u1", "source": str(audio), "target": "gold and gold",
                  "target_hotword_ids": ["gold"], "entities": mentions}
        aligned = [
            {**source, "source_utt_id": "u1", "aligned_mention_id": mention["mention_id"],
             "aligned_hotword_id": "gold", "hotword_start_sec": start, "hotword_end_sec": end}
            for mention, (start, end) in zip(mentions, ((1.8, 2.2), (4.5, 4.7)))
        ]
        return source, aligned

    def test_grouping_completeness_and_first_occurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            source, aligned = self.fixtures(Path(directory))
            grouped, focused = prepare_timed_records([source], aligned)
            self.assertEqual(len(grouped), 1)
            self.assertEqual(len({row["key"] for row in focused}), 2)
            self.assertEqual([e["is_first_occurrence"] for e in grouped[0]["entities"]], [True, False])
            validate_timed_records(grouped, check_audio=True)
            with self.assertRaisesRegex(ValueError, "missing alignment"):
                prepare_timed_records([source], aligned[:1])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare_timed_records([source], aligned + aligned[:1])
            grouped[0]["audio"] = "other.wav"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_timed_records(grouped)

    def test_prepare_boundary_and_evaluator_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, aligned = self.fixtures(work)
            source_path, aligned_path = work / "source.jsonl", work / "aligned.jsonl"
            source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            aligned_path.write_text("".join(json.dumps(row) + "\n" for row in aligned), encoding="utf-8")
            grouped_path, focus_path = work / "timed.jsonl", work / "focus.jsonl"
            commands = [
                ["scripts/prepare_online_manifest.py", "--source-manifest", str(source_path),
                 "--aligned-manifest", str(aligned_path), "--output", str(grouped_path),
                 "--aligned-output", str(focus_path), "--report", str(work / "prep.json")],
                ["scripts/build_boundary_stress.py", "--manifest", str(focus_path),
                 "--output-dir", str(work / "boundary"), "--output-manifest", str(work / "boundary.jsonl")],
            ]
            for command in commands:
                subprocess.run([sys.executable, *command], cwd=ROOT, check=True, capture_output=True)
            variants = [json.loads(line) for line in (work / "boundary.jsonl").read_text().splitlines()]
            self.assertEqual(len(variants), 14)
            self.assertEqual(len({r["key"] for r in variants}), 14)
            self.assertEqual(len({manifest_source(r) for r in variants}), 14)
            validate_timed_records(variants, check_audio=True)
            for row in variants:
                self.assertEqual(row["source"], row["audio"])
                self.assertAlmostEqual(row["duration_sec"], 6 + row["leading_silence_sec"])
                silence_samples = round(row["leading_silence_sec"] * 16000)
                with wave.open(row["source"], "rb") as audio:
                    self.assertEqual(audio.readframes(silence_samples), b"\0\0" * silence_samples)
                    self.assertEqual(audio.readframes(96000), b"\x01\x00" * 96000)
                for original, shifted in zip(((1.8, 2.2), (4.5, 4.7)), row["entities"]):
                    self.assertAlmostEqual(shifted["start_sec"], original[0] + row["leading_silence_sec"])
                    self.assertAlmostEqual(shifted["end_sec"], original[1] + row["leading_silence_sec"])
            grouped = json.loads(grouped_path.read_text())
            result = {**grouped, **example_record(entities=grouped["entities"]),
                      "audio": source["source"], "source": source["source"],
                      "audio_sha256": grouped["audio_sha256"]}
            result_path = work / "result.jsonl"
            result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            metric_path = work / "metrics.json"
            subprocess.run([sys.executable, "scripts/eval_hotword_retrieval.py",
                            "--input", str(result_path), "--output", str(metric_path),
                            "--online", "--bootstrap-samples", "40"], cwd=ROOT,
                           check=True, capture_output=True)
            metrics = json.loads(metric_path.read_text())
            self.assertEqual(metrics["online"]["by_k"]["50"]["target_count"], 1)
            self.assertAlmostEqual(metrics["online"]["by_k"]["50"]["latency_ms_mean"], 1900)
            self.assertTrue(Path(str(metric_path) + ".entities.jsonl").is_file())
            self.assertTrue(Path(str(metric_path) + ".refreshes.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
