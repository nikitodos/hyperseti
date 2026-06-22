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


def load_real_background(filepath, max_load_gb=2.0, max_tchans=8192):
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
        max_tchans (int or None): if given, loads only the first
                 max_tchans time integrations (via blimpy's t_start/
                 t_stop, which are integration-index, not second,
                 arguments) instead of the whole file. This dataset's
                 files are ~234000 time samples each (60 s at 256 us
                 sampling) -- find_et()'s gulp_size parameter controls
                 FREQUENCY channels per gulp, not time samples, so it
                 does not bound this on its own; without max_tchans, an
                 injection-recovery run that repeats this load per
                 (file x SNR x drift) grid point would reprocess the
                 full 234k-sample array every time, which is wasteful
                 and unnecessarily slow for an experiment that (with
                 the current near-stationary-only drift grid, see
                 calibrate_injection_grid) does not need anywhere near
                 60 s of integration to be meaningful. None means load
                 the whole file (the previous, unbounded behavior).

    Returns:
        frame (setigen.Frame), header (dict): the loaded frame and the
                 raw blimpy header dict (for logging/provenance)
    """
    import blimpy as bl
    import setigen as stg

    if max_tchans is not None:
        wf = bl.Waterfall(filepath, max_load=max_load_gb, t_start=0, t_stop=max_tchans)
    else:
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


def calibrate_injection_grid(header, n_snr_levels=4,
                              drift_grid_hz_s=(0.01,)):
    """ Compute an SNR/drift-rate injection grid for this file.

    IMPORTANT, AND THE REASON THIS FUNCTION WAS REWRITTEN TWICE: an
    earlier version of this function expressed drift rate as a
    FRACTION OF CHANNEL WIDTH PER TIMESTEP (i.e. relative to the
    file's own tsamp). That was wrong for a different reason than the
    one below -- see git history / README changelog for that first
    fix -- and was corrected to use fixed, physically motivated
    absolute Hz/s values instead (Sheikh et al. 2019, arXiv:1910.01148;
    operational ranges of a few to ~50 Hz/s used by Breakthrough
    Listen / COSMIC).

    THIS SECOND ISSUE IS DIFFERENT AND MORE FUNDAMENTAL, AND IS WHY
    THE DEFAULT GRID HERE IS NOW A SINGLE NEAR-ZERO VALUE: this
    dataset's channel width (0.5 MHz, from a backend designed for
    broadband FRB detection, not narrowband SETI spectroscopy) is
    roughly six orders of magnitude coarser than the synthetic
    dataset's (2.79 Hz). Over this file's 60 s duration, even the
    upper end of physically realistic SETI drift rates (~20 Hz/s)
    displaces an injected signal by ~0.0024 channels -- three orders
    of magnitude below one channel, i.e. completely unresolvable by
    any dedoppler search regardless of trial-drift-rate grid
    resolution. This is a property of the instrument this archive was
    recorded with, not a bug in this script or in the dedoppler
    kernel.

    CONSEQUENCE FOR WHAT THIS EXPERIMENT CAN AND CANNOT DEMONSTRATE:
    with this dataset, at this channel resolution, injection-recovery
    here is NOT a test of dedoppler search performance (there is no
    resolvable drift trajectory to search across) -- it is a test of
    detection and KLT-based RFI removal for an effectively STATIONARY
    narrowband signal on a real background. That is still a valid and
    useful experiment (it directly tests whether KLT damages a weak
    real signal the way Section 5 worried about, on real RFI rather
    than synthetic RFI), but it does NOT exercise or validate the
    drift-rate dimension of the pipeline, and must not be reported or
    summarized as if it does. If a dedoppler-capable real-data test is
    needed later, it requires a dataset with channel width at most a
    few Hz to tens of Hz (e.g. a standard Breakthrough Listen
    fine-channelized filterbank), not this archive.

    Args:
        header (dict): blimpy header dict (must contain 'tsamp', 'foff',
                 and, if available, 'nsamples' -- used only for the
                 sanity check below, not to set the grid's scale)
        n_snr_levels (int): number of SNR grid points (same SNR values
                 as the synthetic dataset's grid: 10/20/40/80, for
                 direct comparability of the SNR axis specifically)
        drift_grid_hz_s (tuple[float]): the absolute drift rates (Hz/s)
                 to inject. Defaults to a single near-zero value (0.01
                 Hz/s, not exactly 0.0 to avoid any edge-case division
                 issues in setigen's path calculation) for the reason
                 explained above. Override explicitly if testing a
                 different, finer-channelized dataset.

    Returns:
        snr_grid (list[float]), drift_grid_hz_s (list[float])

    Raises:
        ValueError: if any requested drift rate would carry the
                 injected signal across the file's bandwidth within its
                 own duration (or, if duration is unknown because
                 nsamples was unavailable and not independently
                 supplied, this check is skipped with a warning rather
                 than silently assumed safe).
    """
    tsamp = header['tsamp']
    df_hz = abs(header['foff']) * 1e6
    nchans = header['nchans']
    bandwidth_hz = nchans * df_hz

    snr_grid = [10, 20, 40, 80][:n_snr_levels]
    drift_grid_hz_s = list(drift_grid_hz_s)

    nsamples = header.get('nsamples')
    if nsamples:
        duration_s = nsamples * tsamp
        for d in drift_grid_hz_s:
            total_excursion_hz = abs(d) * duration_s
            if total_excursion_hz > bandwidth_hz:
                raise ValueError(
                    f"calibrate_injection_grid: drift_rate={d} Hz/s over "
                    f"this file's duration ({duration_s:.3f} s) would carry "
                    f"the injected signal across {total_excursion_hz/1e6:.3f} MHz, "
                    f"exceeding the file's {bandwidth_hz/1e6:.3f} MHz bandwidth. "
                    f"Lower drift_grid_hz_s or restrict the injection to a "
                    f"shorter sub-segment of the file."
                )
    else:
        logger.warning(
            "calibrate_injection_grid: nsamples unavailable in header, "
            "cannot verify the requested drift grid stays within the "
            "file's bandwidth over its duration. Proceeding without this "
            "sanity check -- verify manually if results look suspicious."
        )

    logger.info(f"calibrate_injection_grid: tsamp={tsamp*1e6:.2f}us, df={df_hz:.4f}Hz, "
                f"bandwidth={bandwidth_hz/1e6:.1f}MHz -> drift grid (Hz/s, fixed, "
                f"NOT scaled to tsamp): {drift_grid_hz_s}")
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
                  device='cpu', margin_fraction=0.1, seed=None, max_tchans=8192):
    """ For one real background file: load it once, then for every
    (snr, drift) grid point, make a FRESH COPY of the loaded background
    (injection must not accumulate across grid points within the same
    file), inject one signal, run both the raw and KLT-cleaned pipeline,
    and record recovery.

    max_tchans is passed through to load_real_background -- see that
    function's docstring for why this dataset's full files (~234k time
    samples) need an explicit bound here, since gulp_size in find_et()
    only bounds frequency channels, not time samples.

    Returns:
        list[dict]: one row per (snr, drift) grid point for this file
    """
    from copy import deepcopy
    from hyperseti.pipeline import find_et

    base_frame, header = load_real_background(filepath, max_tchans=max_tchans)
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
                # max_dd is forced to EXACTLY 0.0 here, not a small
                # nonzero value -- this was found necessary by actually
                # running this function (an earlier version used
                # max(drift_grid_hz_s) * 1.5, which crashed with
                # "RuntimeError: No steps!"). The reason: hyperseti's
                # dedoppler() has a special case (dedoppler.py) that
                # returns a single drift=0 trial when max_dd == 0 and
                # min_dd is None; any other value, however small, falls
                # through to plan_stepped()'s general trial-grid
                # construction, which rounds N_dopp_upper/lower to
                # integer channel counts via int(max_dd / delta_dd) --
                # at this dataset's channel resolution, ANY physically
                # realistic drift rate (see calibrate_injection_grid's
                # docstring) rounds to a degenerate [0, 0] range there,
                # which plan_stepped cannot build a non-empty grid from.
                # This is consistent with, not a workaround for, the
                # "near-stationary-only, not a dedoppler test" framing
                # established above: searching a single drift=0 trial IS
                # the correct search for this experiment, not an
                # approximation of a wider one.
                config_raw = {
                    'preprocess': {'normalize': True},
                    'dedoppler': {'kernel': 'dedoppler', 'max_dd': 0.0,
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
                        # Tolerance is deliberately generous on the
                        # drift axis: with max_dd forced to 0.0 above,
                        # the search has exactly one trial (drift=0),
                        # so any near-stationary injection (drift up to
                        # whatever calibrate_injection_grid's default
                        # produces, e.g. 0.01 Hz/s) must match against
                        # that single trial. Using a tolerance smaller
                        # than the injected drift itself (e.g. the old
                        # 0.1 * max(drift_grid_hz_s) = 0.001 Hz/s for a
                        # 0.01 Hz/s injection) would systematically fail
                        # to match every true positive.
                        drift_diff = (hits['drift_rate'] - drift).abs()
                        match = (chan_diff <= margin // 2) & (drift_diff <= max(drift_grid_hz_s) + 1e-6)
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
    parser.add_argument('--max-tchans', type=int, default=8192,
                         help="Max time samples loaded per file (this dataset's full files are "
                              "~234000 samples / 60s -- gulp_size does not bound this, see "
                              "load_real_background docstring). Use a smaller value for a faster "
                              "first test, e.g. 2048.")
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
                                 args.klt_var_frac, args.klt_window, device=args.device,
                                 max_tchans=args.max_tchans)
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
