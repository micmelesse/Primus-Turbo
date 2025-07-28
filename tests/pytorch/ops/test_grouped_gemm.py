import torch

from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref


def test_blockwise_fp8_grouped_gemm_func(B, M, N, K, ori_dtype):
    device = "cuda"
    seg_lens = torch.zeros([B], dtype=torch.int32, device=device)
    seg_lens[0] = 256
    seg_lens[1] = 768
    seg_lens[2] = 768
    seg_lens[3] = 256
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=False)
    b = torch.randn((B, K, N), dtype=ori_dtype, device=device, requires_grad=False)
    out = grouped_gemm(a, b, seg_lens, 0)
    out_ref = grouped_gemm_ref(a.clone(), b.clone(), seg_lens.clone(), False)
    print(out)
    print(out_ref)


if __name__ == "__main__":
    torch.manual_seed(1234)
    test_blockwise_fp8_grouped_gemm_func(4, 512, 1024, 2048, torch.float16)
