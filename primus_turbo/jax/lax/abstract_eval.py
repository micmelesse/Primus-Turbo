from functools import partial
from typing import Any, Dict

from jax.core import Primitive, ShapedArray

from .primitives import multiply_add_p
from .utils import _lax_implements

ABSTRACT_EVAL_TABLE: Dict[Primitive, Any] = {}
abstract_eval_implements = partial(_lax_implements, ABSTRACT_EVAL_TABLE)


def _elementwise_abstract_eval(args, kwargs):
    assert len(args) > 0, "elementwise like primitive has no input."
    x = args[0]
    for y in args[1:]:
        assert x.shape == y.shape
    return ShapedArray(x.shape, x.dtype)


@abstract_eval_implements([multiply_add_p])
def elementwise_impls(_, *args, **kwargs):
    return _elementwise_abstract_eval(args, kwargs)
