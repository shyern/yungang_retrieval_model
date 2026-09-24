import unittest

import torch

from mmmrag.chinese_clip_local import (
    build_token_weight_predictor,
    combine_global_local_scores,
    compact_text_tokens,
    late_interaction_scores,
    predict_token_weights,
    symmetric_contrastive_loss,
    valid_text_token_mask,
)


class ChineseCLIPLocalAlignmentTests(unittest.TestCase):
    def test_special_tokens_are_excluded(self):
        input_ids = torch.tensor([[101, 11, 12, 102, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
        mask = valid_text_token_mask(input_ids, attention_mask, (0, 101, 102))
        self.assertEqual(mask.tolist(), [[False, True, True, False, False]])

    def test_compaction_retains_beginning_and_end(self):
        tokens = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
        mask = torch.ones((1, 6), dtype=torch.bool)
        compact, compact_mask = compact_text_tokens(tokens, mask, max_tokens=3)
        self.assertEqual(compact.flatten().tolist(), [0.0, 2.0, 5.0])
        self.assertTrue(compact_mask.all())

    def test_late_interaction_prefers_matching_pair(self):
        text_tokens = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
        )
        image_patches = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
        )
        mask = torch.ones((2, 1), dtype=torch.bool)
        scores = late_interaction_scores(text_tokens, image_patches, mask)
        self.assertTrue(torch.equal(scores.argmax(dim=1), torch.tensor([0, 1])))
        self.assertGreater(scores[0, 0].item(), scores[0, 1].item())

    def test_local_loss_and_score_blend_are_finite(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        self.assertTrue(torch.isfinite(symmetric_contrastive_loss(logits)))
        global_scores = torch.tensor([[0.8, 0.2]])
        local_scores = torch.tensor([[0.4, 0.6]])
        blended = combine_global_local_scores(global_scores, local_scores, 1.0)
        self.assertTrue(torch.allclose(blended, torch.tensor([[0.6, 0.4]])))

    def test_token_weights_are_masked_and_mean_normalized(self):
        torch.manual_seed(7)
        predictor = build_token_weight_predictor(hidden_size=4, bottleneck=3)
        hidden = torch.randn(2, 5, 4)
        mask = torch.tensor([[True, True, False, True, False], [True, False, True, True, False]])
        weights = predict_token_weights(predictor, hidden, mask)
        self.assertTrue(torch.all(weights[~mask] == 0))
        self.assertTrue(torch.allclose(
            weights.sum(dim=1), mask.sum(dim=1, dtype=weights.dtype), atol=1e-5
        ))


if __name__ == "__main__":
    unittest.main()
