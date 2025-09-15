import torch

from primus_turbo.pytorch.core import TurboStream


def test_turbo_stream():
    device = "cuda:0"
    turbo_stream = TurboStream(device=device, cu_masks=[0xFFFFFFFF])

    x = torch.ones(10, device=device)
    y = torch.ones(10, device=device)
    out = torch.zeros_like(x)

    with torch.cuda.stream(turbo_stream.torch_stream):
        out = x + y

    turbo_stream.torch_stream.synchronize()
    assert torch.all(out == 2)
