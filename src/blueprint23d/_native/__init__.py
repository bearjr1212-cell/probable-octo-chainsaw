"""Compiled kernels for the hot paths.

Each module here is an optional accelerator with a pure-Python counterpart
of identical semantics. Nothing in the package may *require* one to be
present: import them through the dispatchers (e.g. ``exact.BACKEND``)
rather than directly, so an uncompiled install still works.
"""
