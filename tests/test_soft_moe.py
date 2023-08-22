import random
from functools import partial

import pytest
import torch
from einops import rearrange
from timm.models.vision_transformer import Attention
from torch import nn

from soft_moe_pytorch import SoftMoELayerWrapper
from soft_moe_pytorch.soft_moe import softmax


def test_softmax():
    """
    Test between custom multi-dim softmax and naive impl.
    """
    for _ in range(20):
        # Single-dim
        x = torch.randn(2, 10, 10)
        y1 = softmax(x, dim=-1)
        y2 = torch.softmax(x, dim=-1)
        assert y1.size() == y2.size()
        assert torch.all(torch.isclose(y1, y2))

        # Multi-dim
        x = torch.randn(2, 10, 10, 10)
        y1 = softmax(x, dim=(2, 3))
        y2 = rearrange(
            x.flatten(start_dim=2).softmax(dim=-1), "b m (n p) -> b m n p", n=10
        )
        assert y1.size() == y2.size()
        assert torch.all(torch.isclose(y1, y2))


def test_soft_moe_layer_forward():
    """
    Test forward with different layers
    """
    for num_experts in [1, 4]:
        for slots_per_experts in [1, 2]:
            for dim in [16, 128]:
                f = SoftMoELayerWrapper(
                    dim=dim,
                    slots_per_expert=slots_per_experts,
                    num_experts=num_experts,
                    layer=nn.Linear,
                    in_features=dim,
                    out_features=32,
                )
                n = random.randint(1, 128)
                inp = torch.randn(1, n, dim)
                out = f(inp)
                assert list(out.shape) == [1, n, 32]
                assert not torch.isnan(out).any(), "Output included NaNs"

    for num_experts in [1, 4]:
        for slots_per_experts in [1, 2]:
            for dim in [16, 128]:
                f = SoftMoELayerWrapper(
                    dim=dim,
                    slots_per_expert=slots_per_experts,
                    num_experts=num_experts,
                    layer=partial(Attention, dim=dim),
                )
                n = random.randint(1, 128)
                inp = torch.randn(1, n, dim)
                out = f(inp)
                assert list(out.shape) == [1, n, dim]
                assert not torch.isnan(out).any(), "Output included NaNs"


def test_soft_moe_layer_input_wrong_features_channels():
    """
    Test for error when input has wrong feature dim
    """
    f = SoftMoELayerWrapper(
        dim=128,
        slots_per_expert=1,
        num_experts=16,
        layer=nn.Linear,
        in_features=128,
        out_features=32,
    )

    with pytest.raises(AssertionError):
        inp = torch.randn(1, 16, 64)
        f(inp)


def test_soft_moe_layer_input_wrong_dim():
    """
    Test for error when input is not 3-dim
    """
    f = SoftMoELayerWrapper(
        dim=128,
        slots_per_expert=1,
        num_experts=16,
        layer=nn.Linear,
        in_features=128,
        out_features=32,
    )

    with pytest.raises(AssertionError):
        inp = torch.randn(1, 16, 64, 64)
        f(inp)

    with pytest.raises(AssertionError):
        inp = torch.randn(1, 16)
        f(inp)
