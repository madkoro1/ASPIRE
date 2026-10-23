"""Loss normalization test for Eq. (2); requires torch (skipped on CPU-only CI without it)."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class TokenSumLossTests(unittest.TestCase):
    def test_accumulated_token_normalization(self):
        import torch
        from aspire.training.train import token_sum_loss
        torch.manual_seed(0)
        logits = [torch.randn(1, 5, 7), torch.randn(1, 3, 7)]
        labels = [torch.tensor([[-100, -100, 1, 2, 3]]), torch.tensor([[-100, 4, 5]])]
        n = sum(int(l[..., 1:].ne(-100).sum()) for l in labels)          # 3 + 2 supervised tokens
        accumulated = sum(token_sum_loss({"logits": x}, y, n) for x, y in zip(logits, labels))
        flat_logits = torch.cat([x[0, :-1] for x in logits])
        flat_labels = torch.cat([y[0, 1:] for y in labels])
        expected = torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        torch.testing.assert_close(accumulated, expected)               # token mean over the whole update


if __name__ == "__main__":
    unittest.main()
