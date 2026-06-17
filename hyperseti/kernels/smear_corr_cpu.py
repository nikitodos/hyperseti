"""
smear_corr_cpu.py -- numpy equivalent of the smear_corr CUDA RawKernel.

N_smear (channels of smearing) is always >= 0 by construction (it comes
from np.abs(...).astype('int32') upstream), so unlike the dedoppler
kernel's `shift[d] * t / T` (which can be negative), the `N_smear / 2`
integer division here has no truncation-vs-floor ambiguity: Python's `//`
and C's `/` agree for non-negative operands. No special-casing needed.
"""
import numpy as np

from ..log import get_logger

logger = get_logger('hyperseti.kernels.smear_corr_cpu')


def smear_corr_kernel_cpu(idata: np.ndarray, N_chan_smear: np.ndarray, F: int, D: int) -> np.ndarray:
    """ CPU equivalent of smear_corr_kernel (CUDA RawKernel).

    Args:
        idata (np.ndarray): (D, F) float32 input array (one beam slice)
        N_chan_smear (np.ndarray): (D,) int32 smearing width per drift trial
        F (int): number of frequency channels
        D (int): number of dedoppler trials

    Returns:
        odata (np.ndarray): (D, F) float32 smear-corrected array
    """
    idata = np.ascontiguousarray(idata, dtype=np.float32)
    odata = np.empty((D, F), dtype=np.float32)

    f = np.arange(F, dtype=np.int64)

    for d in range(D):
        N_smear = int(N_chan_smear[d])
        row = idata[d, :]

        if N_smear > 1:
            half = N_smear // 2  # safe: N_smear >= 0 always, see module docstring
            in_range = (f + half < F) & (f - half > 0)

            # Moving sum of width N_smear centered at each valid f,
            # matching the CUDA loop `for i in 0..N_smear: movsum +=
            # idata[idx + i - N_smear/2]`.
            out_row = row.copy()
            if np.any(in_range):
                valid_f = f[in_range]
                # offsets i - half for i in [0, N_smear)
                offsets = np.arange(N_smear, dtype=np.int64) - half
                gather_idx = valid_f[:, None] + offsets[None, :]  # (n_valid, N_smear)
                movsum = row[gather_idx].sum(axis=1)
                out_row[valid_f] = movsum / np.sqrt(np.float32(N_smear))
            odata[d, :] = out_row
        else:
            odata[d, :] = row

    return odata
