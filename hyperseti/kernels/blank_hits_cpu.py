"""
blank_hits_cpu.py -- numpy equivalent of the blank_hits CUDA RawKernel.

Unlike the dedoppler kernel, this one has no `t / T` integer division in
its index calculation (idx = cidx[tid] + F*t + shift[tid]*t), so there is
no C-truncation-vs-Python-floor subtlety to replicate here. The logic is
otherwise a direct port: for each hit, zero out its drifting track plus
padding channels on either side, guarding against out-of-range indices
exactly as the CUDA kernel's `if (idx < F*T && idx > 0)` checks do.
"""
import numpy as np

from ..log import get_logger

logger = get_logger('hyperseti.kernels.blank_hits_cpu')


def blank_hits_kernel_cpu(data: np.ndarray, cidx: np.ndarray, shift: np.ndarray,
                           N_pad_lower: np.ndarray, N_pad_upper: np.ndarray,
                           F: int, T: int) -> np.ndarray:
    """ CPU equivalent of blank_hits_kernel (CUDA RawKernel).

    Modifies `data` in place (matching the CUDA kernel's behaviour of
    writing directly into the buffer it's given) and also returns it for
    convenience.

    Args:
        data (np.ndarray): flat (T*F,) or (T, F) float32 array to blank
                            in place (single beam/polarization slice)
        cidx (np.ndarray): (B,) int32 start channel index per hit
        shift (np.ndarray): (B,) int32 per-hit channel shift per timestep
        N_pad_lower (np.ndarray): (B,) int32 lower padding bound (negative)
        N_pad_upper (np.ndarray): (B,) int32 upper padding bound (positive)
        F (int): number of frequency channels
        T (int): number of time steps

    Returns:
        data (np.ndarray): the same array, blanked in place
    """
    data_flat = data.reshape(-1)  # view, not copy -- in-place writes propagate
    B = cidx.shape[0]
    t_arr = np.arange(T, dtype=np.int64)

    for b in range(B):
        # idx[t] = cidx[b] + F*t + shift[b]*t  (no integer division here)
        idx = int(cidx[b]) + F * t_arr + int(shift[b]) * t_arr  # (T,)

        valid = (idx < F * T) & (idx > 0)
        idx_valid = idx[valid]
        if idx_valid.size > 0:
            data_flat[idx_valid] = 0.0

        N_up = int(N_pad_upper[b])
        N_lo = int(N_pad_lower[b])

        # Upper padding: p = 1 .. N_up-1, guard idx+p < F*T
        if N_up > 1:
            p_range = np.arange(1, N_up, dtype=np.int64)
            idx_up = idx[:, None] + p_range[None, :]            # (T, N_up-1)
            valid_up = (idx_up < F * T)
            # idx itself can be invalid (e.g. < 0) at some t; the CUDA
            # kernel only reaches the padding writes after the outer
            # `if (idx < F*T && idx > 0)` guard for the central write,
            # so padding is only applied where the central idx was valid.
            valid_up &= valid[:, None]
            data_flat[idx_up[valid_up]] = 0.0

        # Lower padding: p = -1 .. N_lo+1 (N_lo is negative), guard idx+p > 0
        if N_lo < -1:
            p_range = np.arange(-1, N_lo, -1, dtype=np.int64)
            idx_lo = idx[:, None] + p_range[None, :]            # (T, |N_lo|-1)
            valid_lo = (idx_lo > 0)
            valid_lo &= valid[:, None]
            data_flat[idx_lo[valid_lo]] = 0.0

    return data
