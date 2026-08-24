"""Optional PyTorch tests for Qwen projector reuse and GLCLAP loss."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from torch import nn

    from asr.contextual.glclap_loss import glclap_loss, multi_positive_contrastive_loss
    from asr.contextual.glclap_model import (
        GLCLAPRetrieverModel,
        QwenGLCLAPEncoder,
        module_parameter_hash,
    )


if HAS_TORCH:

    class _FakeAudioTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(4, 896)
            self.ln_post = nn.LayerNorm(896)
            self.proj1 = nn.Linear(896, 896)
            self.proj2 = nn.Linear(896, 1024)
            self.forward_calls = 0

        def forward(self, input_features, feature_lens=None):
            del feature_lens
            self.forward_calls += 1
            hidden = self.ln_post(self.backbone(input_features))
            output = self.proj2(torch.nn.functional.gelu(self.proj1(hidden)))
            return SimpleNamespace(last_hidden_state=output)


    class _FakeThinker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_tower = _FakeAudioTower()
            self.model = SimpleNamespace(embed_tokens=nn.Embedding(32, 1024))

        def get_input_embeddings(self):
            return self.model.embed_tokens


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class GLCLAPModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.thinker = _FakeThinker()
        self.encoder = QwenGLCLAPEncoder(self.thinker)

    def test_projector_copy_and_epoch_lr_schedule(self) -> None:
        model = GLCLAPRetrieverModel(self.encoder, mode="qwen_projector_warmstart")
        self.assertTrue(torch.equal(model.retrieval_projector.proj1.weight, self.thinker.audio_tower.proj1.weight))
        self.assertIsNot(model.retrieval_projector.proj1.weight, self.thinker.audio_tower.proj1.weight)
        self.assertFalse(any(parameter.requires_grad for parameter in model.retrieval_projector.parameters()))
        groups = model.optimizer_parameter_groups(3e-4, 3e-5)
        self.assertEqual([group["lr"] for group in groups], [3e-4, 3e-5])
        model.set_epoch(1)
        self.assertTrue(all(parameter.requires_grad for parameter in model.retrieval_projector.parameters()))

    def test_frozen_qwen_hash_and_gradients_do_not_change(self) -> None:
        model = GLCLAPRetrieverModel(self.encoder, mode="qwen_post_projector_frozen")
        before = module_parameter_hash(self.thinker)
        features = self.encoder.extract_audio_features(torch.randn(5, 4), torch.tensor([5]))
        audio = model.encode_audio_features(features)
        text = model.encode_text_tokens(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))
        loss = audio.square().mean() + text.square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.thinker.parameters()))
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups())
        optimizer.step()
        self.assertEqual(before, module_parameter_hash(self.thinker))

    def test_packed_audio_batch_uses_one_tower_call_and_matches_serial(self) -> None:
        first = torch.randn(3, 4)
        second = torch.randn(5, 4)
        lengths = torch.tensor([3, 5])
        serial = self.encoder.extract_audio_features_batch(
            [first, second], lengths, packed=False
        )
        serial_calls = self.thinker.audio_tower.forward_calls
        packed = self.encoder.extract_audio_features_batch(
            [first, second], lengths, packed=True
        )
        self.assertEqual(serial_calls, 2)
        self.assertEqual(self.thinker.audio_tower.forward_calls, 3)
        self.assertEqual([tuple(item.post_projector.shape) for item in packed], [(3, 1024), (5, 1024)])
        for expected, actual in zip(serial, packed):
            self.assertTrue(torch.allclose(expected.pre_projector, actual.pre_projector))
            self.assertTrue(torch.allclose(expected.post_projector, actual.post_projector))

    def test_training_forward_exposes_all_ddp_trainables(self) -> None:
        model = GLCLAPRetrieverModel(self.encoder, mode="qwen_post_projector_frozen")
        features = self.encoder.extract_audio_features(torch.randn(5, 4), torch.tensor([5]))
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        audio, transcripts, hotwords, temperature = model(
            [features],
            input_ids,
            attention_mask,
            input_ids,
            attention_mask,
        )
        self.assertEqual(tuple(audio[0].shape), (5, 512))
        self.assertEqual(tuple(transcripts.shape), (1, 512))
        self.assertEqual(tuple(hotwords.shape), (1, 512))
        (audio[0].sum() + transcripts.sum() + hotwords.sum() + temperature).backward()
        self.assertIsNotNone(model.log_temperature.grad)


    def test_global_only_loss_can_skip_local_similarity(self) -> None:
        audio = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
        transcripts = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        output = glclap_loss(
            audio,
            transcripts,
            torch.empty((0, 8)),
            torch.eye(2, dtype=torch.bool),
            torch.empty((2, 0), dtype=torch.bool),
            temperature=torch.tensor(0.07),
            local_weight=0.0,
            compute_local=False,
        )
        self.assertEqual(tuple(output.local_logits.shape), (2, 0))
        self.assertEqual(float(output.local_loss), 0.0)

    def test_global_local_loss_and_multi_positive_mask(self) -> None:
        audio = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1).requires_grad_()
        transcripts = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1).requires_grad_()
        hotwords = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1).requires_grad_()
        global_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
        local_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        output = glclap_loss(
            audio,
            transcripts,
            hotwords,
            global_mask,
            local_mask,
            temperature=torch.tensor(0.07),
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        value = multi_positive_contrastive_loss(torch.zeros((2, 2)), global_mask)
        self.assertAlmostEqual(float(value), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
