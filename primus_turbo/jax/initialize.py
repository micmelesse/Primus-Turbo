import jax
from jax.interpreters import batching, mlir

from primus_turbo.jax.lax.abstract_eval import ABSTRACT_EVAL_TABLE
from primus_turbo.jax.lax.batching import BATCHING_TABLE
from primus_turbo.jax.lax.impl import IMPL_TABLE
from primus_turbo.jax.lax.lowering import LOWERING_TABLE


def initialize():
    """the function of jax_plugins entry_point"""
    from primus_turbo.jax._C import registrations

    print("Primus-Turbo register ffi target...")

    for name, target in registrations().items():
        jax.ffi.register_ffi_target(name, target, platform="ROCM")

    # register lowering for primitives
    for primitive, func in LOWERING_TABLE.items():
        mlir.register_lowering(primitive, func, platform="rocm")

    # regisger impl for primitives
    for primitive, func in IMPL_TABLE.items():
        primitive.def_impl(func)

    # register abstract eval for primitives
    for primitive, func in ABSTRACT_EVAL_TABLE.items():
        primitive.def_abstract_eval(func)

    # register batching for primitives
    for primitive, func in BATCHING_TABLE.items():
        batching.primitive_batchers[primitive] = func
