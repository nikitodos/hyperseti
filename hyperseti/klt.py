"""
klt.py -- xp-agnostic (numpy or cupy) Karhunen-Loeve Transform for
RFI removal and adaptive noise estimation.

PROVENANCE: this module is vendored (copied, not imported as an external
dependency) from nikitodos/seti_klt's klt/core.py, ported from the
original klt() in 03a_clean_filterbank.py. It is duplicated here rather
than imported from seti_klt as a package dependency because the two
repos are currently developed in parallel and seti_klt is not published
to PyPI; if/when seti_klt is packaged, this file should be replaced with
an actual dependency on it rather than kept as a manual copy, to avoid
the two diverging.

Original docstring follows.
---

Two distinct outputs, on purpose (see conversation history / paper
methods section): RFI removal and noise estimation are separate
mechanisms built on the same eigendecomposition, not one thing. Keeping
them as two return values (rather than folding noise estimation
silently into the cleaning step) is what makes the ablation in
validation/run_ablation.py meaningful -- the SNR improvement from each
can be measured independently.

1. RFI removal (matches original semantics exactly):
       R = cov(signals - mean(signals))
       eigenspectrum, eigenvectors = eigh(R), sorted descending
       neig = number of eigenvectors needed to reach `threshold` of
              cumulative variance
       rfi_template = reconstruction using only the top `neig`
                      eigenvectors (the original code's `recsignals`)
       cleaned = signals - abs(rfi_template)
   The abs() in the subtraction is preserved exactly as in the original
   `output_matrix = (matrix - np.abs(rfitemplate_full.T))` -- this is a
   deliberate choice in the original code (avoids adding energy back in
   where the low-rank reconstruction locally overshoots negative), not a
   default introduced here.

2. Noise variance estimation (new, distinct from RFI removal):
       noise_var = mean of the *discarded* eigenvalues (those below the
                   variance threshold), normalized by the number of
                   channels in the window.
   This is what GulpPipeline.preprocess() in the hyperseti fork can use
   as a per-subband noise estimate, instead of normalize()'s single
   scalar std for the whole gulp. It is computed from the discarded
   eigenvalues specifically because those represent the variance NOT
   explained by the dominant (RFI) modes -- i.e. an estimate of the
   noise floor after the dominant correlated structure is accounted for.
"""
import numpy as np


def eigenbasis(matrix, xp=np):
    """ Eigendecomposition of a symmetric covariance matrix, sorted by
    decreasing eigenvalue.

    Args:
        matrix: (N, N) covariance matrix
        xp: array module (numpy or cupy). Both expose linalg.eigh with
            the same signature, so this is a near-free port.

    Returns:
        eigenspectrum, eigenvectors: eigenvalues (descending) and the
            corresponding eigenvectors (columns), same array module as
            the input.
    """
    eigenvalues, eigenvectors = xp.linalg.eigh(matrix)

    if eigenvalues[0] < eigenvalues[-1]:
        eigenvalues = xp.flipud(eigenvalues)
        eigenvectors = xp.fliplr(eigenvectors)
    eigenspectrum = eigenvalues
    return eigenspectrum, eigenvectors


def count_elements_for_threshold(arr, threshold, xp=np):
    """ Number of (descending-sorted) elements needed to reach `threshold`
    fraction of the total sum. """
    sorted_arr = xp.sort(arr)[::-1]
    total_sum = xp.sum(sorted_arr)
    cumulative_sum = xp.cumsum(sorted_arr)
    num_elements = xp.searchsorted(cumulative_sum, threshold * total_sum, side='right') + 1
    return num_elements


def klt(signals, threshold, xp=np):
    """ Karhunen-Loeve Transform: low-rank RFI template + reconstruction.

    Direct xp-agnostic port of the original klt() in
    03a_clean_filterbank.py. Same inputs, same outputs, same semantics --
    see module docstring for what changed (dispatch only, not the math).

    Args:
        signals: (N_time, N_chan) array, channel-major covariance (i.e.
                 covariance computed between channels, across time) --
                 matches the original convention where the caller passes
                 datagrabbed.T (time as rows).
        threshold (float): cumulative variance fraction to retain in the
                 low-rank reconstruction (this is what upstream calls
                 var_frac).
        xp: array module (numpy or cupy)

    Returns:
        neig (int): number of eigenvectors retained
        eigenspectrum: eigenvalues, descending
        eigenvectors: corresponding eigenvectors
        recsignals: low-rank reconstruction (the RFI template -- NOT yet
                    subtracted from signals; that's done by
                    klt_denoise() below, matching how the original
                    process_data() did the subtraction itself rather
                    than inside klt())
    """
    R = xp.cov((signals - xp.mean(signals, axis=0)), rowvar=False)

    eigenspectrum, eigenvectors = eigenbasis(R, xp=xp)

    neig = count_elements_for_threshold(eigenspectrum, threshold, xp=xp)

    mean_signals = xp.mean(signals, axis=0)
    coeff = xp.matmul((signals[:, :] - mean_signals), xp.conjugate(eigenvectors[:, :]))
    neig_int = int(neig)
    recsignals = xp.matmul(coeff[:, 0:neig_int], xp.transpose(eigenvectors[:, 0:neig_int])) + mean_signals

    return neig, eigenspectrum, eigenvectors, recsignals


def klt_denoise(data, var_frac, klt_window, xp=np, estimate_noise=True):
    """ Apply windowed KLT cleaning (and optionally noise estimation) to
    a (N_time, N_chan) array, in-memory -- this is the entry point meant
    for use inside hyperseti's GulpPipeline.preprocess(), as opposed to
    the original script's file-level process_data().

    Args:
        data: (N_time, N_chan) array (e.g. a hyperseti gulp/DataArray
              slice for one beam)
        var_frac (float): cumulative variance threshold (same meaning as
              `threshold` in klt())
        klt_window (int): number of channels per KLT covariance window
        xp: array module (numpy or cupy)
        estimate_noise (bool): if True, also compute a per-window noise
              variance estimate from the discarded eigenvalues (see
              module docstring, mechanism 2). If False, only the cleaned
              array is returned (noise_var is None) -- this lets the
              ablation in validation/run_ablation.py isolate "RFI
              removal only" from "RFI removal + noise estimation".

    Returns:
        cleaned: (N_time, N_chan) array, same shape as input
        noise_var: (N_chan,) array of per-channel noise variance
              estimates (each channel gets its window's estimate
              broadcast across the window), or None if
              estimate_noise=False
    """
    import math

    N_time, N_chan = data.shape
    nchunks = math.ceil(N_chan / klt_window)

    cleaned = xp.zeros_like(data, dtype=xp.result_type(data.dtype, xp.float64))
    noise_var = xp.zeros(N_chan, dtype=xp.float64) if estimate_noise else None

    for ii in range(nchunks):
        c_start = ii * klt_window
        c_end = min((ii + 1) * klt_window, N_chan)

        window = data[:, c_start:c_end].astype(xp.float64)
        # klt() expects (N_time, N_window_chan) with covariance computed
        # between channels (rowvar=False in np.cov), which is exactly
        # window's own orientation here -- no transpose needed (unlike
        # the original script, which read column-major .fil blocks and
        # so transposed before calling klt()).
        neig, eigenspectrum, eigenvectors, rfi_template = klt(window, var_frac, xp=xp)

        # RFI removal: same subtraction as the original script's
        # `matrix - np.abs(rfitemplate_full.T)`, just without the extra
        # transpose this layout doesn't need.
        cleaned[:, c_start:c_end] = window - xp.abs(rfi_template)

        if estimate_noise:
            # Noise variance estimate: mean of the *discarded* eigenvalues
            # (those beyond neig), i.e. the variance not explained by the
            # dominant (presumed-RFI) modes. Normalized by window size so
            # this is a per-channel variance, not a window-total one.
            n_window_chan = c_end - c_start
            neig_int = int(neig)
            if neig_int < n_window_chan:
                discarded = eigenspectrum[neig_int:]
                window_noise_var = xp.mean(discarded) / n_window_chan
            else:
                # Degenerate case: every eigenvector was kept (var_frac
                # essentially 1.0, or a tiny window). No discarded
                # component to estimate noise from; fall back to the
                # smallest retained eigenvalue as a conservative proxy
                # rather than returning zero (which would silently
                # claim "no noise" downstream).
                window_noise_var = eigenspectrum[-1] / n_window_chan
            noise_var[c_start:c_end] = window_noise_var

    return cleaned, noise_var
