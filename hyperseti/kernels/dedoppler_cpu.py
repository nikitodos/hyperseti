"""
dedoppler_cpu.py -- numpy-vectorized equivalents of the CUDA RawKernels in
kernels/dedoppler.py.

These functions must reproduce the *exact* per-element semantics of the
CUDA C kernels, not just their general intent, in particular:

1. Integer division in the index calculation
   ---------------------------------------
   The CUDA kernel computes:
       idx = tid + (F * t) + (shift[d] * t / T)
   where `shift[d] * t` and `T` are C `int`s. C integer division truncates
   towards zero, e.g. -7 / 2 == -3 in C. Python's `//` operator instead
   floors towards -infinity, e.g. -7 // 2 == -4. Since `shift[d]` can be
   negative (negative drift trials), these two are NOT interchangeable --
   using `//` would silently produce a different (wrong) index for roughly
   half of the negative-drift trials, while looking completely correct for
   positive-drift trials and small test cases that happen not to exercise
   the off-by-one boundary. We replicate C truncation explicitly with
   np.trunc / fixed-point integer arithmetic (see _c_int_div below).

2. Boundary masking
   ----------------
   The CUDA kernel guards every access with `if (idx < F*T && idx > 0)`,
   i.e. out-of-range contributions are simply skipped (treated as adding
   zero), not wrapped around or raising. We replicate this with an
   explicit boolean mask rather than direct fancy indexing, which would
   otherwise wrap negative indices (numpy semantics) instead of skipping
   them (CUDA kernel semantics here).

3. The DDSK formula
   -----------------
   (N*T+1)/(T-1) * (T*(S2/S1**2) - 1) is the generalized spectral kurtosis
   estimator of Nita & Gary (2010b) with shape parameter d=1 (appropriate
   for the Gaussian-noise assumption used here), with T playing the role
   of the outer accumulation count M and N the inner per-timestep
   accumulation count. This is correct as written in the original CUDA
   kernel; the CPU port preserves it unchanged.
"""
import numpy as np

from ..log import get_logger

logger = get_logger('hyperseti.kernels.dedoppler_cpu')


def _c_int_div(a: np.ndarray, b: int) -> np.ndarray:
    """ C-style integer division (truncation towards zero), vectorized.

    Equivalent to C's `a / b` for int a, b. NOT equivalent to Python's
    `a // b`, which floors towards -infinity for mixed-sign operands.

    Args:
        a (np.ndarray): integer array (numerator)
        b (int): integer scalar (denominator), assumed != 0 by caller
    Returns:
        np.ndarray: a / b with C truncation semantics, same shape as a
    """
    # np.trunc operates on floats; cast back to the integer dtype the
    # caller expects (the kernel's `idx` is declared as `int`).
    return np.trunc(a.astype(np.float64) / b).astype(np.int64)


def dedoppler_kernel_cpu(data: np.ndarray, shift: np.ndarray, F: int, T: int) -> np.ndarray:
    """ CPU equivalent of dedoppler_kernel (CUDA RawKernel) in kernels/dedoppler.py.

    Args:
        data (np.ndarray): flat (T*F,) or (T, F) float32 array, time-major
                            (matches the CUDA kernel's flat (T x F) layout)
        shift (np.ndarray): (D,) int32 array of per-trial channel shifts
        F (int): number of frequency channels
        T (int): number of time steps

    Returns:
        dedopp (np.ndarray): (D, F) float32 array of dedoppler-summed data,
            normalized by sqrt(T) exactly as the CUDA kernel does.
    """
    data_flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
    D = shift.shape[0]
    dedopp = np.zeros((D, F), dtype=np.float32)

    tid = np.arange(F, dtype=np.int64)            # (F,)
    t_arr = np.arange(T, dtype=np.int64)           # (T,)

    for d in range(D):
        s = int(shift[d])
        # idx[t, tid] = tid + F*t + trunc(s*t / T)   -- C truncation, see _c_int_div
        offset_t = _c_int_div(s * t_arr, T)         # (T,)  -- one value per timestep
        idx = tid[None, :] + (F * t_arr)[:, None] + offset_t[:, None]  # (T, F)

        valid = (idx < F * T) & (idx > 0)
        idx_safe = np.where(valid, idx, 0)          # placeholder index, masked out below
        contrib = np.where(valid, data_flat[idx_safe], 0.0)

        dd_val = contrib.sum(axis=0)                # sum over t, shape (F,)
        dedopp[d, :] = dd_val / np.sqrt(np.float32(T))

    return dedopp


def dedoppler_kurtosis_kernel_cpu(data: np.ndarray, shift: np.ndarray, F: int, T: int, N: int) -> np.ndarray:
    """ CPU equivalent of dedoppler_kurtosis_kernel (CUDA RawKernel).

    See module docstring for the DDSK formula provenance.

    Args:
        data (np.ndarray): flat or (T, F) float32 array
        shift (np.ndarray): (D,) int32 array of per-trial channel shifts
        F, T (int): n_chan, n_time
        N (int): number of accumulations averaged per timestep (N_acc)

    Returns:
        dedopp_sk (np.ndarray): (D, F) float32 spectral kurtosis array
    """
    data_flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
    D = shift.shape[0]
    dedopp_sk = np.zeros((D, F), dtype=np.float32)

    tid = np.arange(F, dtype=np.int64)
    t_arr = np.arange(T, dtype=np.int64)

    for d in range(D):
        s = int(shift[d])
        offset_t = _c_int_div(s * t_arr, T)
        idx = tid[None, :] + (F * t_arr)[:, None] + offset_t[:, None]

        valid = (idx < F * T) & (idx > 0)
        idx_safe = np.where(valid, idx, 0)
        vals = np.where(valid, data_flat[idx_safe], 0.0)

        S1 = vals.sum(axis=0)
        S2 = (vals ** 2).sum(axis=0)

        # (N*T+1)/(T-1) * (T*(S2/S1^2) - 1) -- matches CUDA kernel exactly,
        # including its behaviour of dividing by S1**2 even where S1==0
        # (produces inf/nan there, same as the original; not silently
        # patched here since that would change the kernel's documented
        # output rather than just porting it).
        with np.errstate(divide='ignore', invalid='ignore'):
            dedopp_sk[d, :] = (N * T + 1) / (T - 1) * (T * (S2 / (S1 ** 2)) - 1)

    return dedopp_sk


def dedoppler_with_kurtosis_kernel_cpu(data: np.ndarray, shift: np.ndarray, F: int, T: int, N: int):
    """ CPU equivalent of dedoppler_with_kurtosis_kernel (CUDA RawKernel).

    Computes both the plain dedoppler sum and the DDSK in one pass, as the
    CUDA kernel does (single loop accumulating S1/S2, two outputs).

    Returns:
        (dedopp, dedopp_sk): tuple of (D, F) float32 arrays
    """
    data_flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
    D = shift.shape[0]
    dedopp = np.zeros((D, F), dtype=np.float32)
    dedopp_sk = np.zeros((D, F), dtype=np.float32)

    tid = np.arange(F, dtype=np.int64)
    t_arr = np.arange(T, dtype=np.int64)
    Tf = np.float32(T)

    for d in range(D):
        s = int(shift[d])
        offset_t = _c_int_div(s * t_arr, T)
        idx = tid[None, :] + (F * t_arr)[:, None] + offset_t[:, None]

        valid = (idx < F * T) & (idx > 0)
        idx_safe = np.where(valid, idx, 0)
        vals = np.where(valid, data_flat[idx_safe], 0.0)

        S1 = vals.sum(axis=0)
        S2 = (vals ** 2).sum(axis=0)

        dedopp[d, :] = S1 / np.sqrt(Tf)
        with np.errstate(divide='ignore', invalid='ignore'):
            dedopp_sk[d, :] = (N * T + 1) / (T - 1) * (T * (S2 / (S1 ** 2)) - 1)

    return dedopp, dedopp_sk
