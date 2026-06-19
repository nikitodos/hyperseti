"""
validation/generate_test_set.py -- generates the synthetic validation
dataset for the RFI removal / noise estimation ablation.

Design (matches the architecture agreed in conversation):
    1. Pure-noise base frame (no RFI, no injected signals).
    2. N_rfi variants with synthetic stationary RFI injected (drift_rate
       ~ 0, strong amplitude) -- the simplest controllable proxy for
       "RFI that should be removed by KLT but would not be removed by
       brute-force dedoppler alone", before attempting real recorded RFI.
    3. For each RFI variant, inject M synthetic ET signals at a grid of
       (snr, drift_rate) via setigen, recording ground truth (start
       frequency, drift rate, requested SNR, AND the locally-measured
       off-source SNR via get_stdoff -- reused as-is from
       05a_signal_injector.py, not reimplemented) into a parallel
       JSON/CSV file.

This intentionally builds the setigen.Frame from scratch (stg.Frame(...))
rather than going through setigen_injection()'s file-based interface
(05a_signal_injector.py's setigen_injection() expects an already-loaded
.fil header/frame), since the validation set has no real .fil input to
start from. get_stdoff is reused directly, not reimplemented, since it
is exactly the right "off-source local noise" estimator for ground-truth
SNR and was already written for this purpose.
"""
import os
import json
import itertools
import logging

import numpy as np
import setigen as stg
from astropy import units as u

logger = logging.getLogger('generate_test_set')


def get_stdoff(x, chan, wchan):
    """ Off-source standard deviation estimator: same function as
    get_stdoff() in 05a_signal_injector.py, copied here verbatim rather
    than imported, because importing that module pulls in `your` (a
    file-writer dependency for .fil output) which is unrelated to this
    function and not needed for synthetic dataset generation. If
    05a_signal_injector.py's get_stdoff ever changes, this copy should
    be updated to match -- it is intentionally NOT reimplemented with
    different logic, just relocated to avoid an unnecessary heavy import.
    """
    mask = np.ones(x.shape[0], dtype=bool)
    mask[chan - wchan: chan + wchan] = 0
    std = np.std(x[mask])
    return std


# --- Default frame parameters (small enough for fast iteration; scale up
# for the real ablation once the pipeline is validated end to end) -----
DEFAULT_FCHANS = 1024
DEFAULT_TCHANS = 64
DEFAULT_DF = 2.7939677238464355 * u.Hz       # matches typical hyperseti/BL fine-channel resolution
DEFAULT_DT = 18.25361108 * u.s
DEFAULT_FCH1 = 6095.214842353016 * u.MHz

# Default ablation grid (as proposed in the architecture discussion)
DEFAULT_SNR_GRID = [10, 20, 40, 80]
DEFAULT_DRIFT_GRID = [0.1, 1.0, 5.0]   # Hz/s

# RFI injection defaults: strong, ~stationary (near-zero drift) signals,
# the simplest controllable proxy for "RFI that KLT should catch and
# brute-force dedoppler alone would not distinguish from a slow-drift ET
# signal at low drift rates".
DEFAULT_RFI_SNR = 200
DEFAULT_RFI_DRIFT = 0.0  # Hz/s -- stationary by construction


def make_base_frame(fchans=DEFAULT_FCHANS, tchans=DEFAULT_TCHANS,
                     df=DEFAULT_DF, dt=DEFAULT_DT, fch1=DEFAULT_FCH1,
                     seed=None):
    """ Pure-noise base frame, no RFI, no injected ET signals. """
    if seed is not None:
        np.random.seed(seed)
    frame = stg.Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)
    frame.add_noise(x_mean=0, x_std=1, noise_type='gaussian')
    return frame


def inject_et_signal(frame, f_start_chan, snr, drift_rate_hz_s, f_width_hz=10.0):
    """ Inject a single synthetic ET (technosignature-like) CW/drifting
    signal into an existing frame, in place. Mirrors the inline pattern
    already used for ad-hoc testing (constant_path / constant_t_profile /
    gaussian_f_profile), kept here as a reusable function rather than
    setigen_injection()'s file-based interface (no .fil header here).

    Args:
        frame (stg.Frame): frame to inject into (modified in place)
        f_start_chan (int): starting frequency channel index
        snr (float): nominal SNR requested from setigen
        drift_rate_hz_s (float): drift rate in Hz/s
        f_width_hz (float): Gaussian frequency profile width in Hz

    Returns:
        frame (stg.Frame): same frame, signal added
    """
    frame.add_signal(
        stg.constant_path(
            f_start=frame.get_frequency(index=f_start_chan),
            drift_rate=drift_rate_hz_s * u.Hz / u.s,
        ),
        stg.constant_t_profile(level=frame.get_intensity(snr=float(snr))),
        stg.gaussian_f_profile(width=f_width_hz * u.Hz),
        stg.constant_bp_profile(level=1),
    )
    return frame


def measure_local_snr(data, f_start_chan, drift_rate_hz_s, df_hz, dt_s, w_chan=20):
    """ Off-source SNR estimate for ground truth, reusing get_stdoff()
    from 05a_signal_injector.py exactly as written (not reimplemented).

    Follows the signal's drift track rather than averaging at a fixed
    channel: for drift_rate != 0, the signal occupies a different
    channel at each timestep, so averaging data.mean(axis=0) at a fixed
    f_start_chan dilutes the signal almost to nothing for any nonzero
    drift (e.g. for the default grid, a 0.1 Hz/s drift over 64 timesteps
    of 18.25s each moves the signal across ~42 channels at 2.79 Hz
    resolution -- a fixed-channel time average mixes in ~41 channels of
    pure noise). Averaging along the actual track instead is what
    correctly recovers a measured SNR close to what was requested.

    Args:
        data (np.ndarray): (tchans, fchans) frame data (post-injection)
        f_start_chan (int): channel where the signal starts (t=0)
        drift_rate_hz_s (float): drift rate in Hz/s
        df_hz (float): channel width in Hz
        dt_s (float): time sample duration in seconds
        w_chan (int): half-width (in channels) of the exclusion window
                      around the *starting* channel when computing the
                      "off-source" std (matches get_stdoff's `wchan`
                      argument) -- using the start channel as the
                      exclusion center is an approximation when drift is
                      large, but errs on the side of excluding too much
                      rather than too little background, which is the
                      safer direction for a noise estimate.

    Returns:
        peak_snr (float): mean(value along the drift track) / off-source
                  std -- a measured, not nominal, SNR
    """
    tchans, fchans = data.shape
    spectrum_mean = data.mean(axis=0)
    std_off = get_stdoff(spectrum_mean, f_start_chan, w_chan)

    # Channel occupied by the signal at each timestep, following the
    # same constant-drift-rate path setigen used to inject it.
    t_arr = np.arange(tchans)
    chan_track = np.rint(f_start_chan + (drift_rate_hz_s * t_arr * dt_s) / df_hz).astype(int)
    chan_track = np.clip(chan_track, 0, fchans - 1)

    track_values = data[t_arr, chan_track]
    peak = float(np.mean(track_values))

    return float(peak / std_off) if std_off > 0 else float('nan')


def generate_rfi_variant(base_seed, fchans=DEFAULT_FCHANS, tchans=DEFAULT_TCHANS,
                          df=DEFAULT_DF, dt=DEFAULT_DT, fch1=DEFAULT_FCH1,
                          n_rfi_lines=2, rfi_snr=DEFAULT_RFI_SNR,
                          rfi_drift=DEFAULT_RFI_DRIFT, rng=None):
    """ One noise+RFI frame, with n_rfi_lines strong near-stationary
    signals injected at random channels (excluding a margin at the
    edges so later ET injection grid points don't collide with RFI
    channels). Returns the frame AND the list of RFI channels used (so
    the ET injection step can avoid them).
    """
    frame = make_base_frame(fchans, tchans, df, dt, fch1, seed=base_seed)
    if rng is None:
        rng = np.random.default_rng(base_seed)

    margin = fchans // 10
    rfi_channels = rng.choice(
        np.arange(margin, fchans - margin), size=n_rfi_lines, replace=False
    ).tolist()

    for ch in rfi_channels:
        inject_et_signal(frame, int(ch), rfi_snr, rfi_drift, f_width_hz=df.to(u.Hz).value * 2)

    return frame, rfi_channels


def generate_dataset(output_dir, snr_grid=None, drift_grid=None,
                      n_rfi_variants=3, n_rfi_lines_per_variant=2,
                      fchans=DEFAULT_FCHANS, tchans=DEFAULT_TCHANS,
                      df=DEFAULT_DF, dt=DEFAULT_DT, fch1=DEFAULT_FCH1,
                      seed=1234):
    """ Generate the full validation dataset: for each RFI variant, one
    .h5 file per (snr, drift_rate) grid point, each containing:
        - the base noise
        - that variant's RFI lines
        - exactly one ET signal at the grid's (snr, drift_rate)
    plus a single ground_truth.json describing every file.

    One ET signal per file (rather than the whole grid in one file) is
    deliberate: it keeps the ablation's "recovered or not" bookkeeping
    unambiguous -- no need to disentangle which hit corresponds to which
    grid point if a detector partially fails on a subset of signals
    sharing a file.

    Args:
        output_dir (str): directory to write .h5 files and ground_truth.json
        snr_grid (list[float]): SNR values to test (default:
                  DEFAULT_SNR_GRID, as agreed in the architecture)
        drift_grid (list[float]): drift rates in Hz/s (default:
                  DEFAULT_DRIFT_GRID)
        n_rfi_variants (int): number of distinct RFI realizations
        n_rfi_lines_per_variant (int): RFI lines injected per variant
        seed (int): master seed for reproducibility

    Returns:
        ground_truth (list[dict]): same content as written to
                  ground_truth.json, for convenience when calling this
                  programmatically (e.g. from run_ablation.py)
    """
    os.makedirs(output_dir, exist_ok=True)
    snr_grid = snr_grid if snr_grid is not None else DEFAULT_SNR_GRID
    drift_grid = drift_grid if drift_grid is not None else DEFAULT_DRIFT_GRID

    master_rng = np.random.default_rng(seed)
    ground_truth = []

    margin = fchans // 10
    # Reserve a fixed ET injection channel per file (away from edges);
    # RFI channels are drawn excluding this one so they never overlap.
    et_f_start_chan = fchans // 2

    file_idx = 0
    for rfi_variant_idx in range(n_rfi_variants):
        variant_seed = int(master_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(variant_seed)

        for snr, drift in itertools.product(snr_grid, drift_grid):
            file_idx += 1

            # Fresh base + RFI for every grid point (not reused across
            # the grid) so each file is self-contained and independently
            # loadable by run_ablation.py without needing to replay any
            # prior injection state.
            frame, rfi_channels = generate_rfi_variant(
                base_seed=variant_seed + file_idx,  # vary noise realization per file
                fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1,
                n_rfi_lines=n_rfi_lines_per_variant,
                rng=rng,
            )

            # Guard against an (unlikely, but possible with random
            # placement) RFI channel landing inside the ET injection
            # window; regenerate RFI channels if so, rather than
            # silently letting the two overlap.
            attempts = 0
            while any(abs(c - et_f_start_chan) < 5 for c in rfi_channels) and attempts < 10:
                frame, rfi_channels = generate_rfi_variant(
                    base_seed=variant_seed + file_idx + attempts + 1,
                    fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1,
                    n_rfi_lines=n_rfi_lines_per_variant,
                    rng=rng,
                )
                attempts += 1

            inject_et_signal(frame, et_f_start_chan, snr, drift)

            fname = f"variant{rfi_variant_idx:02d}_snr{snr:g}_drift{drift:g}.h5"
            fpath = os.path.join(output_dir, fname)
            frame.save_h5(fpath)

            measured_snr = measure_local_snr(
                frame.data, et_f_start_chan, drift,
                df.to(u.Hz).value, dt.to(u.s).value,
            )

            ground_truth.append({
                'filename': fname,
                'rfi_variant': rfi_variant_idx,
                'rfi_channels': rfi_channels,
                'rfi_snr': DEFAULT_RFI_SNR,
                'rfi_drift_hz_s': DEFAULT_RFI_DRIFT,
                'et_f_start_chan': et_f_start_chan,
                'et_f_start_mhz': float(frame.get_frequency(index=et_f_start_chan)) / 1e6,
                'et_snr_requested': snr,
                'et_snr_measured_local': measured_snr,
                'et_drift_rate_hz_s': drift,
                'fchans': fchans,
                'tchans': tchans,
                'df_hz': df.to(u.Hz).value,
                'dt_s': dt.to(u.s).value,
            })

    gt_path = os.path.join(output_dir, 'ground_truth.json')
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    logger.info(f"generate_dataset: wrote {file_idx} files + ground_truth.json to {output_dir}")
    return ground_truth


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate KLT+dedoppler ablation validation dataset")
    parser.add_argument('-o', '--output-dir', required=True, help="Output directory for .h5 files + ground_truth.json")
    parser.add_argument('--n-rfi-variants', type=int, default=3)
    parser.add_argument('--n-rfi-lines', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    generate_dataset(
        args.output_dir,
        n_rfi_variants=args.n_rfi_variants,
        n_rfi_lines_per_variant=args.n_rfi_lines,
        seed=args.seed,
    )
