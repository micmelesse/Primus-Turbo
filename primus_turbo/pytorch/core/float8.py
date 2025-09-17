###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import get_device_compute_capability

__all__ = ["float8_e4m3", "float8_e5m2"]


def is_fp8_dtype(dtype):
    TORCH_FP8_DTYPE = [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]
    return dtype in TORCH_FP8_DTYPE


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 4):
        return True, ""
    return (
        False,
        "Device compute capability gfx942 or higher required for FP8 execution.",
    )


def check_mxfp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for MXFP8 execution.",
    )


def check_fp8_ocp_support() -> Tuple[bool, str]:
    """Return if fp8 ocp support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP8 OCP format.",
    )


###################################################

try:
    if check_fp8_ocp_support()[0]:
        float8_e4m3 = torch.float8_e4m3fn
        float8_e5m2 = torch.float8_e5m2
    else:
        float8_e4m3 = torch.float8_e4m3fnuz
        float8_e5m2 = torch.float8_e5m2fnuz
except AttributeError:
    raise RuntimeError("Your PyTorch build does not support FP8 types.")

###################################################


class Format(Enum):
    """
    Supported FP8 formats.
    """

    E4M3 = auto()
    E5M2 = auto()
    HYBRID = auto()


class ScalingGranularity(Enum):
    TENSORWISE = auto()
    ROWWISE = auto()
    BLOCKWISE = auto()


class ScalingStrategy(Enum):
    DYNAMIC = auto()
    # DELAYED_SCALING = auto() # TODO: undetermined


class Float8QuantConfig:
    format: Format = None
    granularity: ScalingGranularity = None
    strategy: ScalingStrategy = None
    block_size: Optional[int] = None  # Default: not used for tensorwise/rowwise

    def mxfp8_scaling(self):
        return isinstance(self, MXFP8BlockwiseQuantConfig)

    def current_scaling(self):
        return isinstance(self, TensorwiseQuantConfig) and self.strategy == ScalingStrategy.DYNAMIC

    def block_scaling(self):
        return isinstance(self, BlockwiseQuantConfig)


@dataclass
class TensorwiseQuantConfig(Float8QuantConfig):
    format: Format = Format.E4M3
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    block_size: Optional[int] = None  # Default: not used for tensorwise/rowwise

    def __post_init__(self):
        assert (
            self.granularity == ScalingGranularity.TENSORWISE and self.strategy == ScalingStrategy.DYNAMIC
        ), "Tensorwise only support current scaling now."


@dataclass
class BlockwiseQuantConfig(Float8QuantConfig):
    format: Format = Format.E4M3
    granularity: ScalingGranularity = ScalingGranularity.BLOCKWISE
    block_size: int = 128  # Override: block_size required for blockwise

    def __post_init__(self):
        assert (
            self.granularity == ScalingGranularity.BLOCKWISE and self.block_size is not None
        ), "block_size must be set when granularity is BLOCKWISE"


@dataclass
class MXFP8BlockwiseQuantConfig(BlockwiseQuantConfig):
    format: Format = Format.E4M3
    granularity: ScalingGranularity = ScalingGranularity.BLOCKWISE
    block_size: int = 32

    def __post_init__(self):
        super().__post_init__()

        ret, msg = check_mxfp8_support()
        assert ret, msg

        assert self.block_size == 32, "MXFP8 block size should be 32."
