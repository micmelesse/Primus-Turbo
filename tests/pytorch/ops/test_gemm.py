import pytest
import torch

import primus_turbo.pytorch as pt
from tests.test_utils import get_tolerances


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
def test_gemm(dtype, layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)

    device = "cuda"
    m, n, k = 512, 2048, 1024

    if layout == "NN":
        a = torch.randn((m, k), dtype=dtype, device=device)
        b = torch.randn((k, n), dtype=dtype, device=device)
        a_ref = a.detach().clone()
        b_ref = b.detach().clone()

        c_ref = torch.matmul(a_ref, b_ref)
    elif layout == "TN":
        a = torch.randn((k, m), dtype=dtype, device=device)
        b = torch.randn((k, n), dtype=dtype, device=device)
        a_ref = a.detach().clone()
        b_ref = b.detach().clone()

        c_ref = torch.matmul(a_ref.T, b_ref)
    else:
        a = torch.randn((m, k), dtype=dtype, device=device)
        b = torch.randn((n, k), dtype=dtype, device=device)
        a_ref = a.detach().clone()
        b_ref = b.detach().clone()

        c_ref = torch.matmul(a_ref, b_ref.T)

    # primus_turbo GEMM
    c = pt.ops.gemm(a, b, dtype, layout)

    # Check close
    torch.testing.assert_close(c, c_ref, **get_tolerances(dtype))
