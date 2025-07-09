from __future__ import annotations

import jax
import numpy as np
from absl.testing import absltest
from jax._src import config
from jax._src import test_util as jtu

from primus_turbo.jax.lax import multiply_add

config.parse_flags_with_absl()


class LaxTest(jtu.JaxTestCase):
    """Numerical tests for LAX operations."""

    @jtu.sample_product(
        input_shape=[(4, 1024), (1, 512)],
        dtype=[np.float32, np.float64, np.complex64, np.complex128],
    )
    def test_multiply_add(self, input_shape, dtype):
        rng = jtu.rand_default(self.rng())
        x = rng(input_shape, dtype)
        y = rng(input_shape, dtype)
        z = rng(input_shape, dtype)

        out = multiply_add(x, y, z)
        base_out = jax.lax.add(jax.lax.mul(x, y), z)

        self.assertArraysAllClose(base_out, out, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
