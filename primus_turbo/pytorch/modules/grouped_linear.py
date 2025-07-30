import math

import torch
import torch.nn as nn

from primus_turbo.pytorch.ops import grouped_gemm

__all__ = ["GroupedLinear"]


class GroupedLinear(torch.nn.Module):
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
        return grouped_gemm(x, self.weight, seg_lens)

    def extra_repr(self) -> str:
        return f"batch={self.batch},in_features={self.in_features}, out_features={self.out_features}"
