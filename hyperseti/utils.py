import numpy as np
from functools import wraps
from inspect import signature
from astropy.units import Unit, Quantity
import pandas as pd
import copy
import time
from typing import Any, Callable

from .xp_compat import cp, HAS_GPU
from .data_array import DataArray, from_metadata, split_metadata
from .dimension_scale import DimensionScale, TimeScale

current_gpu_id = 0

# Logging
from .log import get_logger
logger      = get_logger('hyperseti.utils')
time_logger = get_logger('hyperseti.timer')

def attach_gpu_device(new_id: int):
    """ On demand, switch to GPU ID new_id.

    No-op on CPU-only systems (HAS_GPU == False): there is no device to
    attach to, so we just log and return rather than raising, since this
    is called unconditionally from find_et()/GulpPipeline regardless of
    the device the pipeline will actually run on.

    Args:
        new_id (int): Integer ID of GPU to bind to
    """
    global current_gpu_id
    if not HAS_GPU:
        logger.debug("attach_gpu_device: No GPU available, skipping (CPU-only mode)")
        return
    try:
        if new_id == current_gpu_id:
            logger.info(f"attach_gpu_device: Already using GPU ({new_id})")
        else: #pragma: no cover
              # (can't run unit test on single-GPU systems)
            cp.cuda.Device(new_id).use()
            logger.info(f"attach_gpu_device: Using device ID ({new_id})")
            current_gpu_id = new_id
    except: #pragma: no cover
        cur_id = cp.cuda.Device().id
        logger.error("attach_gpu_device: attach_gpu_device: cp.cuda.Device({}).use() FAILED!".format(new_id))
        logger.warning("attach_gpu_device: Will continue to use current device ID ({})".format(cur_id))
        raise


def timeme(func: Callable[[], Any]) -> Callable[[], Any]:
    """ Timing decorator 
    
    Usage:
        from log import set_log_level
        set_log_level('hyperseti.timer', 'info')

        @timeme
        def do_something(x):
            return x
    """
    def wrapper(*arg, **kwargs):
        t1 = time.time()
        res = func(*arg, **kwargs)
        tt = time.time() - t1
        time_logger.info(f"Time taken: {func.__name__}, {tt:.3f} s")
        return res
    return wrapper