"""
peak_finder_cpu.py -- numpy equivalents of the two-stage peak-finding
CUDA RawKernels in kernels/peak_finder.py (max_kernel, max_reduce_kernel).

Stage 1 (find_max_1D): for each frequency channel, find the maximum value
and its index along the slow-varying (time/drift) axis.

Stage 2 (find_max_reduce): group channels into contiguous blocks of size K
and find the maximum value (and corresponding indices) within each block.

Both stages are pure reductions with no `t / T`-style integer division, so
unlike the dedoppler kernel there is no C-truncation-vs-Python-floor issue
to worry about here -- this ports directly to reshape + argmax.

Known pre-existing limitation (inherited from upstream, not introduced by
this port): the CUDA kernel implicitly assumes N_chan % K == 0 (see the
PeakFinderMan docstring upstream: "will likely crash if N_chan % K != 0").
The CPU port below raises an explicit error in that case instead of
producing a silently wrong result, but it does NOT attempt to generalize
to non-divisible cases -- that would be a behavior change beyond what was
asked, not just a port.
"""
import numpy as np


def find_max_1d_cpu(data: np.ndarray):
    """ CPU equivalent of max_kernel (stage 1).

    Args:
        data (np.ndarray): (T, F) float32 array, time-major (matches the
                            CUDA kernel's (T x F) layout)

    Returns:
        maxval (np.ndarray): (F,) float32 max value per channel over time
        maxidx (np.ndarray): (F,) int32 time index of that max per channel
    """
    data = np.asarray(data, dtype=np.float32)
    T, F = data.shape

    # CUDA kernel initializes maxval[tid] = 0 and only updates on strictly
    # greater values, so an all-negative column would report max=0 with
    # idx=0 (never updated). Replicate that by seeding the running max at
    # zero rather than at data[0], not at -inf.
    maxval = np.zeros(F, dtype=np.float32)
    maxidx = np.zeros(F, dtype=np.int32)

    col_max = data.max(axis=0)            # (F,)
    col_argmax = data.argmax(axis=0).astype(np.int32)

    beats_zero = col_max > 0
    maxval = np.where(beats_zero, col_max, 0.0).astype(np.float32)
    maxidx = np.where(beats_zero, col_argmax, 0).astype(np.int32)

    return maxval, maxidx


def find_max_reduce_cpu(maxval: np.ndarray, maxidx: np.ndarray, K: int):
    """ CPU equivalent of max_reduce_kernel (stage 2).

    Args:
        maxval (np.ndarray): (F,) float32, output of find_max_1d_cpu
        maxidx (np.ndarray): (F,) int32, output of find_max_1d_cpu
        K (int): block size. F must be divisible by K (see module
                 docstring -- this is a pre-existing upstream constraint,
                 not something introduced by the CPU port).

    Returns:
        maxval_k (np.ndarray): (F // K,) float32, max value per block
        maxidx_f (np.ndarray): (F // K,) int32, channel index of that max
        maxidx_t (np.ndarray): (F // K,) int32, time index of that max
    """
    F = maxval.shape[0]
    if F % K != 0:
        raise RuntimeError(
            f"find_max_reduce_cpu: N_chan ({F}) is not divisible by K ({K}). "
            f"This mirrors a pre-existing constraint in the upstream CUDA "
            f"kernel (see PeakFinderMan docstring); it is not a new "
            f"restriction introduced by the CPU port."
        )
    n_blocks = F // K

    val_blocks = maxval.reshape(n_blocks, K)
    idx_blocks = maxidx.reshape(n_blocks, K)

    local_argmax = val_blocks.argmax(axis=1)              # (n_blocks,)
    maxval_k = val_blocks[np.arange(n_blocks), local_argmax].astype(np.float32)
    maxidx_f = (np.arange(n_blocks) * K + local_argmax).astype(np.int32)
    maxidx_t = idx_blocks[np.arange(n_blocks), local_argmax].astype(np.int32)

    return maxval_k, maxidx_f, maxidx_t
