"""
validation/real_data/inject_and_recover.py -- injection-recovery
experiment on REAL filterbank backgrounds (FRB 121102 / Lovell Telescope,
Zenodo 10.5281/zenodo.3974768), addressing the central weakness of the
synthetic-only ablation: that "zero false positives" there could be an
artifact of injecting RFI that was constructed to be exactly the kind
of low-rank structure KLT is good at removing.

WHAT THIS DOES, AND DOES NOT, DEMONSTRATE
------------------------------------------
Correct framing (do not deviate from this in the paper or in any
write-up of these results):
    "Running the pipeline on real data and counting fewer hits" is NOT
    a valid result on its own -- fewer hits could mean real RFI got
    removed, OR it could mean the KLT stage damaged real astrophysical
    signal, OR it could mean the detector simply became more
    conservative. None of those are distinguishable without ground
    truth. This script does NOT do that comparison and should not be
    used to produce that comparison.

What this DOES provide ground truth for: a synthetic technosignature-
like signal injected on top of the file's own real background (real
noise, real RFI, real instrumental artifacts, whatever they are -- not
yet characterized, see inspect_header.py and the RFI characterization
note below). Recovery of THAT known, injected signal, comparing raw
vs. KLT-cleaned pipelines, is the valid injection-recovery measurement,
directly analogous to what generate_test_set.py / run_ablation.py did
on synthetic backgrounds.

IMPORTANT CAVEATS, STATED EXPLICITLY (do not silently drop these when
reporting results):
    1. This dataset (single FRB pulse detections from a pulsar/transient
       search campaign) was NOT verified to contain RFI representative
       of a SETI dedoppler search's typical false-positive sources.
       It might be relatively RFI-clean, or contain RFI of a
       completely different character (impulsive/broadband, given it's
       a transient-search instrument) than the near-stationary
       narrowband RFI modeled in the synthetic dataset. Whatever RFI
       characterization comes out of running this script is a
       description of THIS dataset's actual interference, not a
       confirmation or refutation of the synthetic dataset's modeling
       choices -- report it as its own finding, not as validation of
       the earlier study.
    2. The native time/frequency resolution of this dataset (per
       Lovell/Jodrell Bank FRB121102 observations of this era: ~800
       channels over 400 MHz, ~256 us sampling -- NOT independently
       verified against these specific files, see inspect_header.py)
       is drastically different from the synthetic dataset (2.79 Hz
       channels, 18.25 s integration). Drift rates and KLT window sizes
       calibrated for one regime are NOT meaningful in the other without
       rescaling. This script computes its dedoppler/KLT parameters
       FROM the file's own header (read via blimpy) rather than reusing
       the synthetic dataset's fixed values -- but the resulting
       absolute drift-rate range tested should be reported explicitly
       as appropriate to THIS regime, not assumed comparable to the
       earlier Hz/s grid.

USAGE:
    python inject_and_recover.py --fil-dir /path/to/extracted/files \
        --output-csv real_data_results.csv [--n-files N] [--seed S]

Requires: blimpy, setigen, hyperseti (CPU-capable fork), numpy, pandas.
Must be run where the tarball was downloaded -- not executable in the
container this script was authored in (zenodo.org is outside that
container's network allowlist; this was written and reasoned about
from the dataset's published metadata and filelist only, never from
direct access to the files themselves).
"""
import os
import sys
import glob
import json
import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('inject_and_recover_real')


def load_real_background(filepath, max_load_gb=2.0):
    """ Load a real filterbank file as a setigen.Frame, preserving its
    actual noise/RFI content and correct frequency-ordering convention.

    Passing the blimpy Waterfall object itself to Frame.from_data (via
    the `waterfall=` kwarg) lets setigen determine the ascending/
    descending frequency convention from the file's own header, rather
    than us guessing or hardcoding it -- this is the documented
    behavior of setigen.Frame.from_data and is the reason this function
    does not set `ascending` manually.

    Args:
        filepath (str): path to a .fil file
        max_load_gb (float): safety cap passed to blimpy; raises rather
                 than silently truncating if a file is unexpectedly huge
                 relative to available RAM (relevant on the 16GB Ryzen
                 environment this is expected to run on first).

    Returns:
        frame (setigen.Frame), header (dict): the loaded frame and the
                 raw blimpy header dict (for logging/provenance)
    """
    import blimpy as bl
    import setigen as stg

    wf = bl.Waterfall(filepath, max_load=max_load_gb)
    header = dict(wf.header)

    data = np.squeeze(wf.data)  # blimpy's Waterfall.data has shape
    # (time_axis=0, beam_axis=1, freq_axis=2) -- verified directly
    # against blimpy's waterfall.py source (self.time_axis/beam_axis/
    # freq_axis attributes, and the np.squeeze(self.data[t0:t1, if_id,
    # f0:f1]) pattern used internally by Waterfall.grab_data), not
    # assumed by analogy. Squeezing the size-1 beam axis is the same
    # pattern blimpy's own grab_data() uses internally.
    if data.ndim != 2:
        raise ValueError(
            f"load_real_background: expected 2D data after squeezing "
            f"polarization axis, got shape {data.shape} for {filepath}. "
            f"This file may have multiple polarizations or an unexpected "
            f"layout -- inspect with inspect_header.py before proceeding "
            f"rather than assuming this function's squeeze is correct here."
        )

    df = abs(header['foff']) * 1e6  # MHz -> Hz, setigen expects Hz-scale Quantity below
    dt = header['tsamp']
    fch1 = header['fch1'] * 1e6  # MHz -> Hz

    from astropy import units as u
    frame = stg.Frame.from_data(
        df=df * u.Hz,
        dt=dt * u.s,
        fch1=fch1 * u.Hz,
        ascending=False,  # placeholder; overwritten by setigen when waterfall= is passed
        data=data.astype(np.float64),
        waterfall=wf,
    )
    return frame, header


def calibrate_injection_grid(header, n_snr_levels=4, n_drift_levels=3,
                              min_drift_fraction_of_band=0.001,
                              max_drift_fraction_of_band=0.02):
    """ Compute an SNR/drift-rate injection grid scaled to THIS file's
    actual time/frequency resolution, rather than reusing the synthetic
    dataset's fixed Hz/s values (which were calibrated for an 18.25 s /
    2.79 Hz regime, not whatever this file turns out to have).

    Drift rates are chosen as a fraction of (channel_width / tsamp) --
    i.e. relative to how many channels per timestep a maximally-fast
    drift would traverse -- so the resulting grid is automatically
    sensible regardless of the file's absolute tsamp/df, rather than
    requiring us to guess Hz/s values appropriate to an unfamiliar
    regime in advance.

    Args:
        header (dict): blimpy header dict (must contain 'tsamp', 'foff')
        n_snr_levels (int): number of SNR grid points (same SNR values
                 as the synthetic dataset's grid: 10/20/40/80, for
                 direct comparability of the SNR axis specifically)
        n_drift_levels (int): number of drift-rate grid points
        min/max_drift_fraction_of_band: bounds on tested drift rate,
                 expressed as channels traversed per timestep (NOT in
                 Hz/s directly), so the grid scales with the file's own
                 resolution

    Returns:
        snr_grid (list[float]), drift_grid_hz_s (list[float]): the
                 calibrated grids, drift values converted to Hz/s for
                 direct use with setigen's constant_path (which accepts
                 either a float-Hz/s value or an explicit Quantity)
    """
    tsamp = header['tsamp']
    df_hz = abs(header['foff']) * 1e6

    snr_grid = [10, 20, 40, 80][:n_snr_levels]  # same SNR values as synthetic grid, by design

    chan_per_timestep_min = min_drift_fraction_of_band
    chan_per_timestep_max = max_drift_fraction_of_band
    drift_fractions = np.linspace(chan_per_timestep_min, chan_per_timestep_max, n_drift_levels)

    drift_grid_hz_s = (drift_fractions * df_hz / tsamp).tolist()

    logger.info(f"calibrate_injection_grid: tsamp={tsamp*1e6:.2f}us, df={df_hz:.4f}Hz "
                f"-> drift grid (Hz/s): {[f'{d:.2f}' for d in drift_grid_hz_s]}")
    return snr_grid, drift_grid_hz_s


def inject_signal(frame, f_start_chan, snr, drift_rate_hz_s, f_width_chan=2):
    """ Inject one synthetic technosignature-like signal into `frame`
    (modified in place), analogous to generate_test_set.py's
    inject_et_signal but with the frequency profile width specified in
    channels rather than a fixed Hz value, since this file's channel
    width is not assumed to match the synthetic dataset's.
    """
    import setigen as stg
    from astropy import units as u

    # frame.df is a bare float already in Hz (verified directly against
    # setigen's source/behavior, not a Quantity requiring .to(u.Hz)) --
    # an earlier version of this function attempted a frame.header
    # fallback that would always fail (frame.header is None for a frame
    # built via from_data), so this accesses frame.df directly instead.
    width_hz = f_width_chan * frame.df

    frame.add_signal(
        stg.constant_path(
            f_start=frame.get_frequency(index=f_start_chan),
            drift_rate=drift_rate_hz_s * u.Hz / u.s,
        ),
        stg.constant_t_profile(level=frame.get_intensity(snr=float(snr))),
        stg.gaussian_f_profile(width=width_hz * u.Hz),
        stg.constant_bp_profile(level=1),
    )
    return frame


def measure_local_snr(data, f_start_chan, drift_rate_hz_s, df_hz, dt_s, w_chan=20):
    """ Same drift-track-following SNR estimator as
    generate_test_set.py's measure_local_snr (kept identical
    deliberately, for consistency between the synthetic and real-data
    ground truth definitions -- see that module for the rationale on
    why this follows the signal's track rather than averaging at a
    fixed channel).
    """
    tchans, fchans = data.shape
    spectrum_mean = data.mean(axis=0)

    mask = np.ones(fchans, dtype=bool)
    lo, hi = max(0, f_start_chan - w_chan), min(fchans, f_start_chan + w_chan)
    mask[lo:hi] = False
    std_off = np.std(spectrum_mean[mask]) if mask.any() else np.std(spectrum_mean)

    t_arr = np.arange(tchans)
    chan_track = np.rint(f_start_chan + (drift_rate_hz_s * t_arr * dt_s) / df_hz).astype(int)
    chan_track = np.clip(chan_track, 0, fchans - 1)
    track_values = data[t_arr, chan_track]
    peak = float(np.mean(track_values))

    return float(peak / std_off) if std_off > 0 else float('nan')


def run_one_file(filepath, snr_grid, drift_grid_hz_s, klt_var_frac, klt_window,
                  device='cpu', margin_fraction=0.1, seed=None):
    """ For one real background file: load it once, then for every
    (snr, drift) grid point, make a FRESH COPY of the loaded background
    (injection must not accumulate across grid points within the same
    file), inject one signal, run both the raw and KLT-cleaned pipeline,
    and record recovery.

    Returns:
        list[dict]: one row per (snr, drift) grid point for this file
    """
    from copy import deepcopy
    from hyperseti.pipeline import find_et

    base_frame, header = load_real_background(filepath)
    fchans = header['nchans']
    margin = max(1, int(fchans * margin_fraction))
    f_start_chan = fchans // 2  # fixed injection channel, away from edges by construction

    rows = []
    for snr in snr_grid:
        for drift in drift_grid_hz_s:
            frame = deepcopy(base_frame)
            inject_signal(frame, f_start_chan, snr, drift)

            measured_snr = measure_local_snr(
                frame.data, f_start_chan, drift,
                abs(header['foff']) * 1e6, header['tsamp'],
            )

            tmp_path = filepath + f'.inj_snr{snr}_drift{drift:.3f}.fil'
            frame.save_fil(tmp_path)

            try:
                config_raw = {
                    'preprocess': {'normalize': True},
                    'dedoppler': {'kernel': 'dedoppler', 'max_dd': max(drift_grid_hz_s) * 1.5,
                                  'min_dd': None, 'apply_smearing_corr': False},
                    'hitsearch': {'threshold': 8, 'min_fdistance': max(8, margin // 4)},
                    'pipeline': {'n_boxcar': 1},
                }
                config_klt = deepcopy(config_raw)
                config_klt['preprocess']['klt'] = {
                    'klt_window': klt_window, 'var_frac': klt_var_frac,
                    'apply_cleaning': True, 'estimate_noise': False,
                }

                result_raw = find_et(tmp_path, config_raw, device=device, gulp_size=fchans)
                result_klt = find_et(tmp_path, config_klt, device=device, gulp_size=fchans)

                for label, result in [('raw', result_raw), ('klt', result_klt)]:
                    hits = result.hit_table
                    recovered = False
                    n_fp = 0
                    if len(hits) > 0:
                        chan_diff = (hits['channel_idx'] - f_start_chan).abs()
                        drift_diff = (hits['drift_rate'] - drift).abs()
                        match = (chan_diff <= margin // 2) & (drift_diff <= 0.1 * max(drift_grid_hz_s))
                        recovered = bool(match.any())
                        n_fp = int(len(hits) - match.sum())

                    rows.append({
                        'filename': os.path.basename(filepath),
                        'config': label,
                        'snr_requested': snr,
                        'snr_measured_local': measured_snr,
                        'drift_rate_hz_s': drift,
                        'recovered': recovered,
                        'n_false_positives': n_fp,
                    })
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)  # don't accumulate injected copies on disk

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fil-dir', required=True, help="Directory containing extracted .fil files")
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--n-files', type=int, default=None, help="Limit to first N files (for a quick test before the full run)")
    parser.add_argument('--klt-var-frac', type=float, default=0.3, help="Best value found on synthetic data; NOT assumed valid here without re-checking")
    parser.add_argument('--klt-window', type=int, default=128)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    filepaths = sorted(glob.glob(os.path.join(args.fil_dir, '*.fil')))
    if args.n_files:
        filepaths = filepaths[:args.n_files]
    if not filepaths:
        logger.error(f"No .fil files found in {args.fil_dir}")
        sys.exit(1)

    logger.info(f"Found {len(filepaths)} file(s). Calibrating injection grid from first file's header...")

    import blimpy as bl
    first_header = dict(bl.Waterfall(filepaths[0], load_data=False).header)
    snr_grid, drift_grid_hz_s = calibrate_injection_grid(first_header)

    all_rows = []
    for i, fp in enumerate(filepaths):
        logger.info(f"[{i+1}/{len(filepaths)}] {fp}")
        try:
            rows = run_one_file(fp, snr_grid, drift_grid_hz_s,
                                 args.klt_var_frac, args.klt_window, device=args.device)
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"  FAILED on {fp}: {type(e).__name__}: {e}")
            continue

        # Write incrementally -- same rationale as run_ablation.py:
        # a long real-data run should not lose completed work if
        # interrupted partway through.
        pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)

    logger.info(f"Done. {len(all_rows)} rows written to {args.output_csv}")

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        summary = df.groupby('config').agg(
            recall=('recovered', 'mean'),
            total_fp=('n_false_positives', 'sum'),
            n=('recovered', 'count'),
        )
        print(summary.to_string())


if __name__ == '__main__':
    main()
