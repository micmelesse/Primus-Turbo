from functools import partial
from typing import Any, Dict

from jax.core import Primitive

from .primitives import multiply_add_p
from .utils import _lax_implements

BATCHING_TABLE: Dict[Primitive, Any] = {}
batching_implements = partial(_lax_implements, BATCHING_TABLE)


def _elementwise_batching(primitive, batch_args, batch_dims, kwargs):
    return primitive.bind(*batch_args, **kwargs), batch_dims[0]


@batching_implements([multiply_add_p])
def elementwise_impls(primitive, batch_args, batch_dims, **kwargs):
    return _elementwise_batching(primitive, batch_args, batch_dims, kwargs)
