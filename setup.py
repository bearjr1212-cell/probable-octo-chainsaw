"""Native extension build.

Metadata lives in pyproject.toml; this file exists only to compile the C
kernels. Every extension is marked ``optional``, so a machine without a
compiler still installs and runs -- the pure-Python implementations in
exact.py and friends take over, more slowly but with identical results
(tests/test_native.py pins that equivalence).

The flags are not incidental. Shewchuk's error-free transformations are
only error-free under strict IEEE-754 binary64 semantics:

* ``-ffp-contract=off`` stops the compiler fusing ``a*b + c`` into an FMA.
  An FMA computes the product to infinite precision before adding, which
  silently destroys the rounding residual that two_product exists to
  capture -- the predicate would return a confidently wrong sign.
* ``-fexcess-precision=standard`` stops intermediates being held in x87
  80-bit registers, which breaks the same invariant on 32-bit targets.
* fast-math is never enabled; it permits reassociation, which invalidates
  every bound in the file.
"""

from setuptools import Extension, setup

STRICT_IEEE_FLAGS = [
    "-O2",
    "-ffp-contract=off",
    "-fexcess-precision=standard",
    "-fno-fast-math",
    "-std=c99",
]

setup(
    ext_modules=[
        Extension(
            "blueprint23d._native.predicates",
            sources=["src/blueprint23d/_native/predicates.c"],
            extra_compile_args=STRICT_IEEE_FLAGS,
            optional=True,
        ),
        Extension(
            "blueprint23d._native.cdt",
            sources=["src/blueprint23d/_native/cdt.c"],
            extra_compile_args=STRICT_IEEE_FLAGS,
            optional=True,
        ),
        Extension(
            "blueprint23d._native.fitkernels",
            sources=["src/blueprint23d/_native/fitkernels.c"],
            extra_compile_args=STRICT_IEEE_FLAGS,
            optional=True,
        ),
    ],
)
