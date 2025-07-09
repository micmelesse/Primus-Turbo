from functools import partial
from typing import Any, Dict

from jax.core import Primitive
from jax.ffi import ffi_lowering

from .primitives import multiply_add_p
from .utils import _lax_implements

LOWERING_TABLE: Dict[Primitive, Any] = {}
lowering_implements = partial(_lax_implements, LOWERING_TABLE)


@lowering_implements([multiply_add_p])
def lowering_ffi_impls(primitive: Primitive, *args, **kwargs):
    return ffi_lowering(primitive.name)(*args, **kwargs)
