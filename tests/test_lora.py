import unittest

import torch

from mmmrag.lora import LoRALinear


class LoRATests(unittest.TestCase):
    def test_adapter_starts_as_identity_and_receives_gradients(self):
        torch.manual_seed(3)
        base = torch.nn.Linear(4, 3)
        for parameter in base.parameters():
            parameter.requires_grad = False
        adapter = LoRALinear.build(base, rank=2, alpha=4.0, dropout=0.0)
        inputs = torch.randn(5, 4)
        self.assertTrue(torch.allclose(adapter(inputs), base(inputs)))
        adapter(inputs).sum().backward()
        self.assertIsNotNone(adapter.lora_B.grad)
        self.assertGreater(adapter.lora_B.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
