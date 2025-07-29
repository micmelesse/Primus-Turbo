import math

import torch
import torch.nn as nn


def grouped_gemm_ref(a, b, seg_lens, trans_b=True):
    seg_lens = seg_lens.cpu().numpy()
    out = []
    start = 0
    for i, size in enumerate(seg_lens):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return torch.cat(out)


def grouped_gemm_variable_k_ref(a, b, seg_lens, trans_a=True, trans_b=False):
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."
    seg_lens = seg_lens.cpu().numpy()
    B = len(seg_lens)
    M = a.shape[1]
    N = b.shape[1]
    out = torch.zeros((B, M, N), dtype=a.dtype, device="cuda", requires_grad=False)
    start = 0
    for i, size in enumerate(seg_lens):
        a_tmp = a[start : start + size, :].t()
        b_tmp = b[start : start + size, :]
        out_tmp = a_tmp @ b_tmp
        out[i] = out_tmp
        start += size
    return out


class GroupedLinearRef(torch.nn.Module):
    def __init__(
        self,
        batch: int,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features  # K
        self.out_features = out_features  # N
        self.batch = batch
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((batch, self.out_features, self.in_features), **factory_kwargs)
        )  # [B,N,K]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,  # [B * M, K],
        seg_lens: torch.Tensor,  # [B,] int64
    ) -> torch.Tensor:
        out = grouped_gemm_ref(x, self.weight, seg_lens)
        return out


def generate_seq_len(B, B_M):
    """
    Generate a tensor of shape (B,) where all elements are multiples of 128 and sum to B_M

    Args:
        B: batch size, length of output tensor
        B_M: target sum of all elements

    Returns:
        seq_len: tensor of shape (B,) with elements multiple of 128, sum equals B_M
    """
    # Check if B_M is divisible by 128
    if B_M % 128 != 0:
        raise ValueError(f"B_M ({B_M}) must be divisible by 128")

    # Check if B_M is large enough for B elements (each at least 128)
    if B_M < B * 128:
        raise ValueError(f"B_M ({B_M}) too small for B ({B}) elements, minimum required is {B * 128}")

    # Convert to units of 128
    total_units = B_M // 128

    # Generate random distribution ensuring no zero elements
    if B == 1:
        # Special case when B=1
        seq_len = torch.tensor([B_M], dtype=torch.int64)
    else:
        # Generate B-1 random split points ensuring each segment has at least 1 unit (128)
        # We need to distribute (total_units - B) extra units among B segments
        extra_units = total_units - B

        # Generate random distribution of extra units
        if extra_units == 0:
            # All elements will be 128
            units = torch.full((B,), 1, dtype=torch.int64)
        else:
            # Randomly distribute extra units
            # Generate B-1 random split points in the range [0, extra_units]
            splits = torch.sort(torch.randint(0, extra_units + 1, (B - 1,))).values
            # Add start (0) and end (extra_units) points
            points = torch.cat([torch.tensor([0]), splits, torch.tensor([extra_units])])
            # Calculate differences to get extra unit counts for each segment
            extra_units_per_segment = torch.diff(points)
            # Add 1 base unit to each segment to ensure no zeros
            units = extra_units_per_segment + 1

        # Convert units to actual values (multiply by 128)
        seq_len = units * 128

    return seq_len
