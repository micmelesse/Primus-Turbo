from jax.core import Primitive
from jax.typing import ArrayLike

multiply_add_p = Primitive("multiply_add")


def multiply_add(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
    return multiply_add_p.bind(x, y, z)
