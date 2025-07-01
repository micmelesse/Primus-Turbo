from typing import Optional

import torch

from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingStrategy,
)
from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8_blockwise

__all__ = ["MXLinear"]


class MXLinear(torch.nn.Linear):
    """ """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[Float8QuantConfig] = None,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        if config is None:
            config = Float8QuantConfig(
                dtype=Format.E4M3,
                granularity=ScalingGranularity.BLOCKWISE,
                strategy=ScalingStrategy.DYNAMIC,
                block_size=128,
            )
        self.config = config

    def forward(self, x):
        # TODO send config
        out = gemm_fp8_blockwise(
            x,
            self.weight,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        config: Optional[Float8QuantConfig] = None,
    ):
        if config is None:
            config = Float8QuantConfig(
                dtype=Format.E4M3,
                granularity=ScalingGranularity.BLOCKWISE,
                strategy=ScalingStrategy.DYNAMIC,
                block_size=128,
            )
        assert isinstance(mod, torch.nn.Linear), f"unsupported type(mod) {type(mod)}"
        assert isinstance(config, Float8QuantConfig)
        mod.__class__ = MXLinear
        mod.config = config
        return mod

    def extra_repr(self):
        return f"{super().extra_repr()}, config={getattr(self, 'config', None)}"


# TODO: For tensorwise and rowwise FP8 GEMM
class Float8Linear(torch.nn.Linear):
    pass
