import pytest
import torch
import torch.nn as nn

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.float8 import Format
from primus_turbo.pytorch.modules import MXLinear
from tests.test_utils import compute_snr, get_tolerances


# TODO: ori_dtype torch.float32
@pytest.mark.parametrize("ori_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("format", [Format.E4M3])
def test_mxlinear(ori_dtype, format):
    turbo.float8_e4m3
    device = "cuda:0"

    seq = 512
    in_features = 4096
    out_features = 1024

    x1 = torch.randn((seq, in_features), dtype=ori_dtype, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()

    # Ref
    model = nn.Linear(in_features, out_features).to(device, dtype=ori_dtype)
    out_ref = model(x1)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    x_grad_ref = x1.grad.detach().clone()
    w_grad_ref = model.weight.grad.detach().clone()
    bias_grad_ref = model.bias.grad.detach().clone()

    # Clean Grad
    model.weight.grad.zero_()
    model.bias.grad.zero_()
    x2.grad = None

    # MX
    MXLinear.from_float(model)
    assert isinstance(model, MXLinear)

    out = model(x2)
    out.backward(grad_out)
    x_grad = x2.grad.detach().clone()
    w_grad = model.weight.grad.detach().clone()
    bias_grad = model.bias.grad.detach().clone()

    # Check
    print("fwd")
    print(out)
    print(out_ref)
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    print("dgrad")
    print(x_grad)
    print(x_grad_ref)
    xgrad_snr = compute_snr(x_grad_ref, x_grad)
    print(f"XGrad-SNR: {xgrad_snr:.2f} dB")
    assert xgrad_snr > 20, "xgrad_snr too low"

    print("wgrad")
    print(w_grad)
    print(w_grad_ref)
    wgrad_snr = compute_snr(w_grad_ref, w_grad)
    print(f"WGrad-SNR: {wgrad_snr:.2f} dB")
    assert wgrad_snr > 20, "wgrad_snr too low"

    torch.testing.assert_close(bias_grad, bias_grad_ref, **get_tolerances(ori_dtype))
