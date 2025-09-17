###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.modules import Linear
from tests.test_utils import get_tolerances


@pytest.mark.parametrize("bs", [[], [1], [2], [3, 2]])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("in_features", [256])
@pytest.mark.parametrize("out_features", [512])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_linear_accuracy(bs, seq_len, in_features, out_features, bias, dtype, enable_torch_compile):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if enable_torch_compile and dtype == torch.float32:
        pytest.skip()  # latest rocm 7 env will meet err.

    torch.manual_seed(42)
    device = "cuda"

    primus_linear = Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    torch_linear = torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    with torch.no_grad():
        torch_linear.weight.copy_(primus_linear.weight)
        if bias:
            torch_linear.bias.copy_(primus_linear.bias)

    if enable_torch_compile:
        torch._dynamo.reset()  # Clean compile cache, avoid cache limit.
        primus_linear = torch.compile(primus_linear, fullgraph=True, mode="max-autotune")
        torch_linear = torch.compile(torch_linear, fullgraph=True, mode="max-autotune")

    x1 = torch.randn(*bs, seq_len, in_features, device=device, dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()

    # === Forward ===
    out1 = primus_linear(x1)
    out2 = torch_linear(x2)
    assert torch.allclose(out1, out2, **get_tolerances(dtype))

    # === Backward ===
    grad_output = torch.randn_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)

    assert torch.allclose(x1.grad, x2.grad, **get_tolerances(dtype)), "Forward output mismatch"

    assert torch.allclose(
        primus_linear.weight.grad, torch_linear.weight.grad, **get_tolerances(dtype)
    ), "Weight grad mismatch"

    if bias:
        assert torch.allclose(
            primus_linear.bias.grad, torch_linear.bias.grad, **get_tolerances(dtype)
        ), "Bias grad mismatch"
