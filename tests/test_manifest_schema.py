import unittest

from asr.data.manifest import (
    manifest_key,
    manifest_source,
    manifest_target,
    validate_manifest_schema,
)


class ManifestSchemaTests(unittest.TestCase):
    def test_canonical_key_source_target_record(self):
        record = {
            "key": "BAC009S0002W0122",
            "source": "/data/example.wav",
            "target": "而对楼市成交抑制作用最大的限购",
        }
        self.assertEqual(manifest_key(record), "BAC009S0002W0122")
        self.assertEqual(manifest_source(record), "/data/example.wav")
        self.assertEqual(manifest_target(record), "而对楼市成交抑制作用最大的限购")
        validate_manifest_schema([record])

    def test_legacy_internal_aliases_remain_supported(self):
        record = {"utt_id": "u1", "audio": "u1.wav", "text": "测试"}
        self.assertEqual((manifest_key(record), manifest_source(record), manifest_target(record)), ("u1", "u1.wav", "测试"))
        validate_manifest_schema([record])

    def test_missing_source_has_record_number(self):
        with self.assertRaisesRegex(ValueError, "manifest record 1.*source"):
            validate_manifest_schema([{"key": "u1", "target": "测试"}])


if __name__ == "__main__":
    unittest.main()
