import warnings

import torch

try:
    import primus_turbo.pytorch._C
    import primus_turbo.pytorch.deep_ep
except ImportError:
    warnings.warn("Primus-Turbo C Extention is not built.")

import primus_turbo.pytorch.modules
import primus_turbo.pytorch.ops
from primus_turbo.pytorch.core import float8

float8_e4m3 = float8.float8_e4m3
float8_e5m2 = float8.float8_e5m2
