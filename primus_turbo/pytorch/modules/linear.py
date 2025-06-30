import math
from typing import Union

import torch
import torch.nn as nn

from primus_turbo.pytorch.ops import gemm

__all__ = ["Linear"]


class _Linear(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Union[torch.Tensor, None],
        dtype: torch.dtype,
    ):
        # [M, K] * [N, K] -> [M, N]
        out = gemm(x, weight, dtype, "NT")

        # TODO(ruibzhan): Move bias into gemm.
        if bias is not None:
            out += bias

        ctx.save_for_backward(x, weight, bias)
        ctx.dtype = dtype

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x, weight, bias) = ctx.saved_tensors

        dgrad = torch.empty_like(x)
        # [M, N] * [N, K] -> [M, K]
        dgrad = gemm(grad_out, weight, ctx.dtype, "NN")

        # [M, N] * [M, K] -> [N, K]
        wgrad = torch.empty_like(weight)
        wgrad = gemm(grad_out, x, ctx.dtype, "TN")

        bgrad = None
        if bias is not None:
            bgrad = grad_out.sum(dim=0)

        return (dgrad, wgrad, bgrad, None)


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = (
            x,
            self.weight,
            self.bias,
            self.dtype,
        )

        return _Linear.apply(*args)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        )
