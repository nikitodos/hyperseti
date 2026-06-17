"""
xp_compat.py -- CPU/GPU array dispatch layer for hyperseti.

This module is the single point where CuPy is imported. Every other module
in the package should import `xp_module`, `get_xp`, `asnumpy`, `HAS_GPU`,
etc. from here instead of doing `import cupy as cp` directly.

Design:
    - If CuPy + a CUDA device are available, HAS_GPU is True and `cp` refers
      to the real cupy module.
    - If not, HAS_GPU is False and `cp` is a thin alias for numpy, so legacy
      code that still says `cp.something` keeps working on CPU as long as
      that "something" exists in numpy too. This is a transitional shim,
      not a long-term substitute for explicit `xp` dispatch.
    - New / modified code should call `get_xp(data_array_or_array)` to find
      out which array module produced a given array, and use that module
      explicitly rather than relying on the shim.

Why this exists:
    Before this module, every file in hyperseti did `import cupy as cp` at
    module level, which means the package could not even be imported on a
    machine without CUDA -- let alone run a CPU-only pipeline. This module
    makes CuPy an optional dependency.
"""
import importlib
import logging
import os

logger = logging.getLogger('hyperseti.xp_compat')

import numpy as np

# --- Detect CuPy + working CUDA device --------------------------------
HAS_GPU = False
_cupy_mod = None

_force_cpu = os.environ.get('HYPERSETI_FORCE_CPU', '0') == '1'

if not _force_cpu:
    try:
        _cupy_mod = importlib.import_module('cupy')
        # Importing cupy can succeed even with no usable device (e.g. driver
        # mismatch, no GPU present). Touch the device count to be sure.
        _ndevices = _cupy_mod.cuda.runtime.getDeviceCount()
        if _ndevices > 0:
            HAS_GPU = True
        else:
            logger.warning(
                "xp_compat: cupy import succeeded but no CUDA devices were "
                "found (getDeviceCount() == 0). Falling back to CPU."
            )
    except Exception as e:  # ImportError, CUDA driver errors, etc.
        logger.info(
            f"xp_compat: cupy not usable ({type(e).__name__}: {e}). "
            f"Running in CPU-only mode."
        )
        HAS_GPU = False
else:
    logger.info("xp_compat: HYPERSETI_FORCE_CPU=1 set, running in CPU-only mode.")

# `cp` is kept as a name for backward compatibility with code that hasn't
# been migrated to explicit xp dispatch yet. On CPU-only systems this is
# literally numpy, so `cp.ndarray`, `cp.asarray`, etc. still resolve, but
# anything cupy-exclusive (RawKernel, cuda.Device, ElementwiseKernel...)
# will raise AttributeError if accessed under CPU-only mode. Those call
# sites must be migrated explicitly -- see kernels/dedoppler_cpu.py and
# kernels/peak_finder_cpu.py for the patterns used in this fork.
if HAS_GPU:
    cp = _cupy_mod
else:
    cp = np


def get_xp(array_or_data_array):
    """ Return the array module (numpy or cupy) that an array belongs to.

    Args:
        array_or_data_array: a numpy/cupy ndarray, or a DataArray-like
            object exposing a `.data` attribute holding such an array.

    Returns:
        module: numpy or cupy
    """
    arr = getattr(array_or_data_array, 'data', array_or_data_array)
    if HAS_GPU and isinstance(arr, _cupy_mod.ndarray):
        return _cupy_mod
    return np


def asnumpy(array):
    """ Convert a numpy or cupy array to numpy, no-op if already numpy. """
    if HAS_GPU and isinstance(array, _cupy_mod.ndarray):
        return _cupy_mod.asnumpy(array)
    return np.asarray(array)


def to_device(array, device='gpu'):
    """ Move an array to the requested device ('gpu' or 'cpu').

    Raises RuntimeError if 'gpu' is requested but no GPU is available --
    deliberately loud, since a silent fallback here would make benchmark
    numbers misleading (you'd think you measured GPU performance and you
    didn't).
    """
    if device == 'gpu':
        if not HAS_GPU:
            raise RuntimeError(
                "to_device: device='gpu' requested but no usable CUDA "
                "device was found. Set device='cpu' or fix the CUDA setup."
            )
        return _cupy_mod.asarray(array)
    elif device == 'cpu':
        return asnumpy(array)
    else:
        raise ValueError(f"to_device: unknown device '{device}', expected 'gpu' or 'cpu'")


def resolve_device(device):
    """ Normalize a requested device string against actual availability.

    'auto' -> 'gpu' if available else 'cpu'.
    Explicit 'gpu' with no GPU available raises, rather than silently
    downgrading to CPU (see to_device docstring for rationale).
    """
    if device == 'auto':
        return 'gpu' if HAS_GPU else 'cpu'
    if device == 'gpu' and not HAS_GPU:
        raise RuntimeError(
            "resolve_device: device='gpu' requested but no usable CUDA "
            "device was found. Use device='cpu' or device='auto'."
        )
    return device
