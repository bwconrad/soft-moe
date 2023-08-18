import pytest
import torch

from soft_moe_pytorch import (soft_moe_vit_base, soft_moe_vit_huge,
                              soft_moe_vit_large, soft_moe_vit_small,
                              soft_moe_vit_tiny)


@pytest.mark.parametrize(
    "model",
    [soft_moe_vit_tiny],
    # [soft_moe_vit_tiny, soft_moe_vit_small, soft_moe_vit_base, soft_moe_vit_large, soft_moe_vit_huge],
)
def test_soft_moe_vit_forward(model):
    """
    Test network forward pass
    """
    for image_size in [128, 224]:
        for in_chans in [1, 3]:
            net = model(
                img_size=image_size,
                in_chans=in_chans,
                num_classes=10,
            )
            net.eval()

            inp = torch.randn(1, in_chans, image_size, image_size)
            out = net(inp)

            assert out.shape[0] == 1
            assert not torch.isnan(out).any(), "Output included NaNs"


@pytest.mark.parametrize(
    "model",
    [soft_moe_vit_tiny],
    # [soft_moe_vit_tiny, soft_moe_vit_small, soft_moe_vit_base, soft_moe_vit_large, soft_moe_vit_huge],
)
def test_soft_moe_vit_backward(model):
    """
    Test network backward pass
    """
    image_size = 224
    num_classes = 10

    net = model(img_size=image_size, num_classes=num_classes)
    num_params = sum([x.numel() for x in net.parameters()])
    net.train()

    inp = torch.randn(1, 3, image_size, image_size)
    out = net(inp)

    out.mean().backward()
    for n, x in net.named_parameters():
        assert x.grad is not None, f"No gradient for {n}"
    num_grad = sum([x.grad.numel() for x in net.parameters() if x.grad is not None])

    assert out.shape[-1] == num_classes
    assert num_params == num_grad, "Some parameters are missing gradients"
    assert not torch.isnan(out).any(), "Output included NaNs"


@pytest.mark.parametrize(
    "model",
    [soft_moe_vit_tiny],
    # [soft_moe_vit_tiny, soft_moe_vit_small, soft_moe_vit_base, soft_moe_vit_large, soft_moe_vit_huge],
)
def test_soft_moe_vit_forward_num_experts(model):
    """
    Test network soft-moe arguments
    """
    image_size = 224
    in_chans = 3
    for num_experts in [1, 4]:
        for slots_per_experts in [1, 2]:
            for moe_layer_index in [6, [0, 2, 10]]:
                net = model(
                    img_size=image_size,
                    in_chans=in_chans,
                    num_classes=10,
                    num_experts=num_experts,
                    slots_per_expert=slots_per_experts,
                    moe_layer_index=moe_layer_index,
                )
                net.eval()

                inp = torch.randn(1, in_chans, image_size, image_size)
                out = net(inp)

                assert out.shape[0] == 1
                assert not torch.isnan(out).any(), "Output included NaNs"
