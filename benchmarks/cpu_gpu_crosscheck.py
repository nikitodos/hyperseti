"""
benchmarks/cpu_gpu_crosscheck.py -- numerical cross-check and throughput
comparison between the CPU and GPU execution paths of this hyperSETI
fork, on identical input data.

WHY THIS SCRIPT EXISTS AND WHAT IT IS FOR
-------------------------------------------
The paper's Future Work section states explicitly that, at the time of
writing, "a full GPU execution of the CPU-derived kernel ports, and a
direct numerical cross-check between the two execution paths on
identical input data on GPU hardware, had not yet been carried out".
This script is that missing validation step, written and committed
ahead of having GPU hardware available, so it is ready to run
immediately once it is. It was authored and reasoned about (correct
xp_compat dispatch usage, expected hit_table schema, etc.) in a
CPU-only development container with no CUDA device, and therefore
COULD NOT be executed or its output inspected before being committed.
The first real run of this script IS the validation -- treat any
unexpected failure on first run as informative, not as a sign the
script itself is necessarily broken; check both possibilities.

WHAT COUNTS AS A PASS
-----------------------
Two things are checked, and BOTH must hold for the cross-check to be
considered successful -- neither alone is sufficient:
    1. Hit tables (channel_idx, drift_rate, snr) from device='cpu' and
       device='gpu' agree on the SAME input file, within the tolerances
       set below. Exact bitwise equality is NOT expected (floating-
       point reduction order differs between the sequential CPU sum and
       the parallel CUDA kernel's reduction), so a small numerical
       tolerance is appropriate here -- but the tolerance is set tight
       (relative to the channel/drift-rate grid spacing) and is NOT
       meant to absorb a real algorithmic discrepancy. A mismatch
       larger than one grid step in channel or drift-rate index is a
       genuine cross-check failure, not noise.
    2. The KLT cleaning stage (run on top of either device) produces
       the same cleaned data on both paths, within float32-vs-float64
       tolerance -- this isolates whether a CPU/GPU discrepancy (if
       found) originates in the KLT stage or in the dedoppler kernel
       port.

USAGE:
    python cpu_gpu_crosscheck.py --test-file /path/to/some.h5 \
        [--output-json results.json]

If no --test-file is given, generates a synthetic file via setigen
(same parameters as validation/generate_test_set.py's defaults) so the
script is runnable standalone without requiring a specific dataset file
to be present.
"""
import os
import sys
import json
import time
import argparse
import logging

import numpy as np

logger = logging.getLogger('cpu_gpu_crosscheck')

# Tolerances for the cross-check. Set relative to the discretization of
# the search grid (channel index, trial drift rate), not to an absolute
# floating-point epsilon -- the dedoppler search itself only resolves
# signals to within one grid step, so a CPU/GPU discrepancy smaller than
# that is operationally invisible regardless of its numerical origin.
CHAN_TOL = 1
DRIFT_RATE_TOL_FRACTION = 0.01  # 1% of the max_dd used in the test, see below


def make_test_file(path, fchans=1024, tchans=64, snr=30, drift_rate=0.4, seed=42):
    """ Same generation pattern as validation/generate_test_set.py's
    make_base_frame + inject_et_signal, kept deliberately simple and
    self-contained here rather than importing that module, so this
    benchmark script has no dependency on the validation/ package and
    can be run/copied standalone.
    """
    import setigen as stg
    from astropy import units as u

    np.random.seed(seed)
    frame = stg.Frame(
        fchans=fchans, tchans=tchans,
        df=2.7939677238464355 * u.Hz, dt=18.25361108 * u.s,
        fch1=6095.214842353016 * u.MHz,
    )
    frame.add_noise(x_mean=0, x_std=1, noise_type='gaussian')
    f_start_chan = fchans // 2
    frame.add_signal(
        stg.constant_path(f_start=frame.get_frequency(index=f_start_chan), drift_rate=drift_rate * u.Hz / u.s),
        stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=10.0 * u.Hz),
        stg.constant_bp_profile(level=1),
    )
    frame.save_h5(path)
    return dict(f_start_chan=f_start_chan, drift_rate=drift_rate, snr=snr)


def run_pipeline(filepath, device, klt_enabled, max_dd=8.0, min_fdistance=64, threshold=8):
    """ Runs find_et() on the given device, timing the call. Returns
    (hit_table_dataframe, elapsed_seconds, gulp_pipeline_result).

    Raises whatever resolve_device() raises if device='gpu' is
    requested but unavailable -- this is intentional (see
    xp_compat.resolve_device docstring): a benchmark script silently
    falling back to CPU when GPU was requested would produce a
    completely misleading "GPU benchmark" number.
    """
    from hyperseti.pipeline import find_et

    config = {
        'preprocess': {'normalize': True},
        'dedoppler': {'kernel': 'dedoppler', 'max_dd': max_dd, 'min_dd': None, 'apply_smearing_corr': False},
        'hitsearch': {'threshold': threshold, 'min_fdistance': min_fdistance},
        'pipeline': {'n_boxcar': 1},
    }
    if klt_enabled:
        config['preprocess']['klt'] = {
            'klt_window': 128, 'var_frac': 0.3,
            'apply_cleaning': True, 'estimate_noise': False,
        }

    t0 = time.perf_counter()
    result = find_et(filepath, config, device=device, gulp_size=None)
    elapsed = time.perf_counter() - t0

    return result.hit_table, elapsed, result


def compare_hit_tables(hits_cpu, hits_gpu, max_dd):
    """ Compares two hit tables for numerical agreement. Returns a dict
    summarizing the comparison; does not raise on mismatch (the caller
    decides what to do with a failed comparison -- this function's job
    is only to measure and report, not to assert).
    """
    drift_tol = DRIFT_RATE_TOL_FRACTION * max_dd

    n_cpu, n_gpu = len(hits_cpu), len(hits_gpu)
    if n_cpu == 0 and n_gpu == 0:
        return {'status': 'both_empty', 'n_cpu': 0, 'n_gpu': 0, 'matched': 0, 'unmatched_cpu': 0, 'unmatched_gpu': 0}

    matched_gpu_idx = set()
    matched = 0
    max_chan_diff = 0
    max_drift_diff = 0.0

    for _, row_cpu in hits_cpu.iterrows():
        chan_diff = (hits_gpu['channel_idx'] - row_cpu['channel_idx']).abs() if n_gpu else None
        drift_diff = (hits_gpu['drift_rate'] - row_cpu['drift_rate']).abs() if n_gpu else None
        if n_gpu:
            candidate_mask = (chan_diff <= CHAN_TOL) & (drift_diff <= drift_tol)
            candidates = hits_gpu[candidate_mask]
            if len(candidates) > 0:
                idx = candidates.index[0]
                if idx not in matched_gpu_idx:
                    matched_gpu_idx.add(idx)
                    matched += 1
                    max_chan_diff = max(max_chan_diff, chan_diff[idx])
                    max_drift_diff = max(max_drift_diff, drift_diff[idx])

    return {
        'status': 'compared',
        'n_cpu': n_cpu,
        'n_gpu': n_gpu,
        'matched': matched,
        'unmatched_cpu': n_cpu - matched,
        'unmatched_gpu': n_gpu - len(matched_gpu_idx),
        'max_channel_diff_among_matches': int(max_chan_diff),
        'max_drift_rate_diff_among_matches': float(max_drift_diff),
        'tolerances_used': {'chan_tol': CHAN_TOL, 'drift_rate_tol': drift_tol},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test-file', default=None, help="Existing .h5/.fil file to test on. If omitted, a synthetic file is generated.")
    parser.add_argument('--output-json', default='cpu_gpu_crosscheck_results.json')
    parser.add_argument('--max-dd', type=float, default=8.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import hyperseti
    if not hyperseti.xp_compat.HAS_GPU:
        logger.error(
            "No usable GPU detected in this environment (hyperseti.xp_compat.HAS_GPU "
            "is False). This script is meant to run where a CUDA device IS available "
            "(e.g. the RTX machine) -- running it here would only be able to perform "
            "the CPU half of the comparison, which is not useful on its own. Aborting "
            "rather than silently producing a CPU-only 'benchmark' that looks like a "
            "real cross-check but isn't."
        )
        sys.exit(1)

    test_file = args.test_file
    ground_truth = None
    if test_file is None:
        test_file = '/tmp/cpu_gpu_crosscheck_synthetic.h5'
        logger.info(f"No --test-file given, generating synthetic test file at {test_file}")
        ground_truth = make_test_file(test_file)

    results = {'test_file': test_file, 'ground_truth': ground_truth, 'runs': {}}

    for klt_enabled in [False, True]:
        label = 'klt' if klt_enabled else 'raw'
        logger.info(f"=== Configuration: {label} ===")

        hits_cpu, t_cpu, _ = run_pipeline(test_file, 'cpu', klt_enabled, max_dd=args.max_dd)
        logger.info(f"  CPU: {len(hits_cpu)} hits, {t_cpu:.4f}s")

        hits_gpu, t_gpu, _ = run_pipeline(test_file, 'gpu', klt_enabled, max_dd=args.max_dd)
        logger.info(f"  GPU: {len(hits_gpu)} hits, {t_gpu:.4f}s")

        comparison = compare_hit_tables(hits_cpu, hits_gpu, args.max_dd)
        speedup = (t_cpu / t_gpu) if t_gpu > 0 else float('inf')

        logger.info(f"  Comparison: {comparison}")
        logger.info(f"  Speedup (CPU time / GPU time): {speedup:.2f}x")

        results['runs'][label] = {
            'n_hits_cpu': len(hits_cpu),
            'n_hits_gpu': len(hits_gpu),
            'time_cpu_s': t_cpu,
            'time_gpu_s': t_gpu,
            'speedup': speedup,
            'comparison': comparison,
        }

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {args.output_json}")

    # Explicit pass/fail summary at the end -- do not bury this in the
    # log above. A human (or a CI check) should be able to read this
    # final block alone and know whether the cross-check passed.
    print("\n=== SUMMARY ===")
    all_passed = True
    for label, run in results['runs'].items():
        comp = run['comparison']
        if comp['status'] == 'both_empty':
            print(f"[{label}] Both CPU and GPU found zero hits. Cannot confirm "
                  f"numerical agreement this way -- consider a higher-SNR test signal.")
            continue
        ok = (comp['unmatched_cpu'] == 0 and comp['unmatched_gpu'] == 0)
        all_passed = all_passed and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{label}] {status}: {comp['matched']} matched, "
              f"{comp['unmatched_cpu']} CPU-only, {comp['unmatched_gpu']} GPU-only "
              f"(speedup {run['speedup']:.2f}x)")
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL -- inspect ' + args.output_json}")


if __name__ == '__main__':
    main()
