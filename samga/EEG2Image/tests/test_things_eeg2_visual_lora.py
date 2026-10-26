from __future__ import annotations

from collections import OrderedDict
import unittest

import torch
from torch import nn

from things_eeg2.visual_lora import (
    FusedQvLoRA, LiveViTFeatures, VisualLoRAConfig, attach_vit_adapters,
)


class SmallViTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(4)
        self.attn = nn.MultiheadAttention(4, 2, batch_first=True)
        self.ln_2 = nn.LayerNorm(4)
        self.mlp = nn.Sequential(OrderedDict((
            ("c_fc", nn.Linear(4, 8)),
            ("activation", nn.GELU()),
            ("c_proj", nn.Linear(8, 4)),
        )))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        return x + attended + self.mlp(self.ln_2(x + attended))


class VisualAdaptationTests(unittest.TestCase):
    def test_default_still_adapts_only_q_and_v(self) -> None:
        blocks = nn.ModuleList([SmallViTBlock()])
        blocks.requires_grad_(False)
        config = VisualLoRAConfig(rank=2, alpha=4.0)
        attach_vit_adapters(blocks, (1,), config)
        live = object.__new__(LiveViTFeatures)
        live.visual = blocks
        names = set(live.adapter_state_dict())
        self.assertEqual(len(names), 4)
        self.assertTrue(all(name.endswith(("q_down", "q_up", "v_down", "v_up")) for name in names))

    def test_all_options_receive_gradients_and_round_trip(self) -> None:
        torch.manual_seed(19)
        blocks = nn.ModuleList([SmallViTBlock()])
        blocks.requires_grad_(False)
        config = VisualLoRAConfig(
            rank=2, alpha=4.0, tune_layer_norm=True,
            lora_key=True, lora_output=True, lora_mlp=True,
        )
        attach_vit_adapters(blocks, (1,), config)
        blocks[0](torch.randn(2, 3, 4)).square().mean().backward()
        live = object.__new__(LiveViTFeatures)
        live.visual = blocks
        state = live.adapter_state_dict()
        self.assertTrue(any(name.endswith("k_up") for name in state))
        self.assertTrue(any("out_proj.parametrizations.weight" in name for name in state))
        self.assertTrue(any("mlp.c_fc.parametrizations.weight" in name for name in state))
        self.assertTrue(any("mlp.c_proj.parametrizations.weight" in name for name in state))
        self.assertTrue(any("ln_1.weight" in name for name in state))
        self.assertTrue(any("ln_2.bias" in name for name in state))
        self.assertTrue(all(parameter.grad is not None for parameter in live.trainable_parameters()))

        restored_blocks = nn.ModuleList([SmallViTBlock()])
        restored_blocks.requires_grad_(False)
        attach_vit_adapters(restored_blocks, (1,), config)
        restored = object.__new__(LiveViTFeatures)
        restored.visual = restored_blocks
        restored.load_adapter_state_dict(state)
        for name, value in restored.adapter_state_dict().items():
            torch.testing.assert_close(value, state[name])

    def test_key_rows_are_unchanged_when_disabled(self) -> None:
        weight = torch.randn(12, 4)
        adapter = FusedQvLoRA(width=4, rank=2, alpha=2.0)
        with torch.no_grad():
            adapter.q_up.fill_(0.25)
            adapter.v_up.fill_(0.25)
        adapted = adapter(weight)
        torch.testing.assert_close(adapted[4:8], weight[4:8])
        self.assertFalse(torch.equal(adapted[:4], weight[:4]))


if __name__ == "__main__":
    unittest.main()
