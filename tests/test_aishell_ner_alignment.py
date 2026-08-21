import unittest

from asr.contextual.types import HotwordEntry
from asr.data.forced_alignment import AlignmentItem, aligned_records_for_source


class AISHELLNERAlignmentTests(unittest.TestCase):
    def test_repeated_gold_mentions_resolve_by_annotated_occurrence(self):
        entries = {"loc-beijing": HotwordEntry(id="loc-beijing", text="北京")}
        record = {
            "utt_id": "u1",
            "audio": "/tmp/u1.wav",
            "text": "北京和北京",
            "target_hotword_ids": ["loc-beijing"],
            "entities": [
                {
                    "mention_id": "u1#entity-00",
                    "hotword_id": "loc-beijing",
                    "text": "北京",
                    "entity_type": "LOC",
                    "char_start": 0,
                    "char_end": 2,
                    "occurrence_index": 0,
                },
                {
                    "mention_id": "u1#entity-01",
                    "hotword_id": "loc-beijing",
                    "text": "北京",
                    "entity_type": "LOC",
                    "char_start": 3,
                    "char_end": 5,
                    "occurrence_index": 1,
                },
            ],
        }
        items = (
            AlignmentItem("北京", 0.1, 0.4),
            AlignmentItem("和", 0.4, 0.5),
            AlignmentItem("北京", 0.5, 0.8),
        )
        outputs = aligned_records_for_source(record, entries, items)
        self.assertEqual(len(outputs), 2)
        self.assertEqual([item["hotword_start_sec"] for item in outputs], [0.1, 0.5])
        self.assertEqual(
            [item["aligned_mention_id"] for item in outputs],
            ["u1#entity-00", "u1#entity-01"],
        )
        self.assertNotEqual(outputs[0]["utt_id"], outputs[1]["utt_id"])


if __name__ == "__main__":
    unittest.main()
