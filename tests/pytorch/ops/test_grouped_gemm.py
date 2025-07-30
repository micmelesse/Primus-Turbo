import pytest
import torch

from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ref.gemm_ref import generate_seq_len, grouped_gemm_ref
from tests.test_utils import get_tolerances


@pytest.mark.parametrize("B", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("M", [256, 512, 2048])
@pytest.mark.parametrize("N_K", [(1024, 2048), (4096, 4096), (4096, 7168)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_blockwise_fp8_grouped_gemm_func(B, M, N_K, dtype):
    N, K = N_K
    device = "cuda"
    seg_lens = generate_seq_len(B, B * M).to(device)  # int64

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # FWD
    out = grouped_gemm(a, b, seg_lens)
    out_ref = grouped_gemm_ref(a_ref, b_ref, seg_lens.clone(), True)
    torch.testing.assert_close(out_ref, out, **get_tolerances(dtype))

    # BWD
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out.backward(grad_out)
    torch.testing.assert_close(a_ref.grad, a.grad, **get_tolerances(dtype))
    torch.testing.assert_close(b_ref.grad, b.grad, **get_tolerances(dtype))
