from functools import partial
from typing import Any, Dict

from jax.core import Primitive
from jax.interpreters import xla

from .primitives import multiply_add_p
from .utils import _lax_implements

IMPL_TABLE: Dict[Primitive, Any] = {}
impl_implements = partial(_lax_implements, IMPL_TABLE)


@impl_implements([multiply_add_p])
def impl_ffi_impls(primitive, *args, **kwargs):
    return xla.apply_primitive(primitive, *args, **kwargs)
