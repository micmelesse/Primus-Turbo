from functools import partial
from typing import Any, Dict, List

from jax.core import Primitive


def _lax_implements(
    global_table: Dict[Primitive, Any],
    primitives: List[Primitive],
):
    """Register primitive impl to table"""

    def decorator(func):
        for p in primitives:
            if not isinstance(p, Primitive):
                raise ValueError()
            global_table[p] = partial(func, p)
        return func

    return decorator
