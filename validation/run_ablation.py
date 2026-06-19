"""
validation/run_ablation.py -- the central experiment: for each file in
the validation dataset (generate_test_set.py), run find_et() in several
preprocessing configurations and compare the resulting hit table against
ground truth.

Configurations tested (as agreed in the architecture):
    (a) raw           -- no KLT
    (b) klt_clean      -- KLT RFI removal only (estimate_noise=False)
    (c) klt_clean_var_frac_scan -- same as (b) but scanning var_frac,
        since the exploratory testing in this conversation found that
        var_frac (not klt_window) is the parameter that determines
        whether a weak ET signal sharing a KLT window with strong RFI
        survives cleaning or gets removed alongside it. This scan is
        what makes that trade-off quantifiable rather than anecdotal.

Noise-estimation-only effects (klt_noise_var feeding into the peak
finder) are NOT exercised by this ablation yet, because hyperseti's
peak_finder does not consume klt_noise_var in the current integration
(see pipeline.py preprocess() comments) -- that is explicitly future
work, not silently assumed to work here.

Matching a found hit to a ground-truth signal: a hit is counted as
"recovered" if its (channel_idx, drift_rate) falls within tolerance of
the ground truth's (et_f_start_chan, et_drift_rate_hz_s). Any other hit
found is counted as a false positive -- in this dataset, that is
overwhelmingly residual/imperfectly-removed RFI, which is itself a
useful number to report (it's the RFI suppression metric, separate from
ET recovery).
"""
import os
import json
import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('run_ablation')


# Tolerance for matching a found hit to the injected ground-truth signal.
# Channel tolerance accounts for the discretization seen in earlier
# testing (e.g. recovering channel 511 for an injection at 512).
# Drift tolerance accounts for the finite drift-trial grid in dedoppler().
DEFAULT_CHAN_TOL = 5
DEFAULT_DRIFT_TOL = 0.05  # Hz/s


def match_hit_to_truth(hit_table, et_f_start_chan, et_drift_rate_hz_s,
                        chan_tol=DEFAULT_CHAN_TOL, drift_tol=DEFAULT_DRIFT_TOL):
    """ Check whether any row in hit_table matches the ground-truth
    injected signal within tolerance.

    Returns:
        recovered (bool)
        measured_snr (float or None): SNR of the matching hit, if found
        n_false_positives (int): number of OTHER hits in hit_table that
            did not match (i.e. everything in hit_table minus the match,
            if any)
    """
    if len(hit_table) == 0:
        return False, None, 0

    chan_diff = (hit_table['channel_idx'] - et_f_start_chan).abs()
    drift_diff = (hit_table['drift_rate'] - et_drift_rate_hz_s).abs()
    match_mask = (chan_diff <= chan_tol) & (drift_diff <= drift_tol)

    n_matches = match_mask.sum()
    n_false_positives = len(hit_table) - n_matches

    if n_matches == 0:
        return False, None, len(hit_table)
    else:
        # If multiple hits match (shouldn't normally happen given
        # min_fdistance grouping, but be defensive), take the
        # highest-SNR one as "the" recovered hit.
        matched_rows = hit_table[match_mask]
        best = matched_rows.loc[matched_rows['snr'].idxmax()]
        return True, float(best['snr']), int(n_false_positives)


def build_configs(max_dd, min_fdistance, threshold, var_frac_scan, klt_window_scan=None):
    """ Build the dict of {config_name: pipeline_config} to test.

    max_dd / min_fdistance must be sized to the dataset being tested
    (see generate_test_set.py's fchans) -- passed in rather than
    hardcoded, since a mismatch raises a clear error in peak_finder_cpu
    (N_chan % K != 0) rather than silently producing wrong results, and
    that error is dataset-size-dependent.

    Args:
        var_frac_scan (iterable[float]): var_frac values to scan
        klt_window_scan (iterable[int] or None): klt_window values to
                  scan. If None, defaults to [256] only (1D scan over
                  var_frac alone, as in the initial small-scale test).
                  If provided, builds the full 2D grid
                  len(var_frac_scan) x len(klt_window_scan).
    """
    configs = {
        'raw': {
            'preprocess': {'normalize': True},
            'dedoppler': {'kernel': 'dedoppler', 'max_dd': max_dd, 'min_dd': None, 'apply_smearing_corr': False},
            'hitsearch': {'threshold': threshold, 'min_fdistance': min_fdistance},
            'pipeline': {'n_boxcar': 1},
        },
    }

    if klt_window_scan is None:
        klt_window_scan = [256]

    for vf in var_frac_scan:
        for kw in klt_window_scan:
            name = f'klt_vf{vf:g}_w{kw}'
            configs[name] = {
                'preprocess': {
                    'normalize': True,
                    'klt': {'klt_window': kw, 'var_frac': vf, 'apply_cleaning': True, 'estimate_noise': False},
                },
                'dedoppler': {'kernel': 'dedoppler', 'max_dd': max_dd, 'min_dd': None, 'apply_smearing_corr': False},
                'hitsearch': {'threshold': threshold, 'min_fdistance': min_fdistance},
                'pipeline': {'n_boxcar': 1},
            }

    return configs


def run_ablation(dataset_dir, output_csv, max_dd=8.0, min_fdistance=64,
                  threshold=8, var_frac_scan=(0.95, 0.7, 0.5, 0.3),
                  klt_window_scan=None, gulp_size=None, device='cpu'):
    """ Run the full ablation over every file in the dataset, for every
    configuration in build_configs(), and write a results CSV.

    Args:
        dataset_dir (str): directory containing the .h5 files and
                  ground_truth.json from generate_test_set.py
        output_csv (str): path to write the results table
        max_dd, min_fdistance, threshold: passed through to find_et();
                  must be consistent with the dataset's fchans (see
                  build_configs docstring)
        var_frac_scan (tuple[float]): var_frac values to test for the
                  klt configurations
        klt_window_scan (tuple[int] or None): klt_window values to scan
                  (2D grid with var_frac_scan); None for 1D scan only
                  (klt_window fixed at 256)
        gulp_size (int or None): if None, uses the dataset's fchans
                  (single-gulp, since these are small synthetic files)
        device (str): 'cpu' or 'gpu'

    Returns:
        results_df (pd.DataFrame): one row per (file, config)
    """
    # Imported here, not at module level, so this script can be
    # `python -m` invoked from anywhere without requiring hyperseti to
    # be on sys.path unless run_ablation() is actually called.
    from hyperseti.pipeline import find_et

    gt_path = os.path.join(dataset_dir, 'ground_truth.json')
    with open(gt_path) as f:
        ground_truth = json.load(f)

    configs = build_configs(max_dd, min_fdistance, threshold, var_frac_scan, klt_window_scan)
    results = []

    # Resume support: if output_csv already has partial results (e.g.
    # from a run that was interrupted -- this can happen if the process
    # is backgrounded and the surrounding shell session ends, which
    # kills orphaned background processes in this environment even with
    # nohup), skip any (filename) whose full set of configs is already
    # present, rather than discarding that work and starting over.
    already_done = set()
    write_header = True
    if os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv)
            if len(existing) > 0:
                write_header = False
                counts = existing.groupby('filename')['config'].nunique()
                already_done = set(counts[counts >= len(configs)].index)
                results = existing.to_dict('records')
                logger.info(f"run_ablation: resuming, {len(already_done)} files already complete "
                            f"({len(existing)} existing rows)")
        except (pd.errors.EmptyDataError, KeyError):
            pass  # empty or header-only file, nothing to resume

    columns = ['filename', 'config', 'rfi_variant', 'snr_requested', 'snr_measured_local',
               'drift_rate_hz_s', 'recovered', 'measured_snr', 'n_false_positives', 'error']
    if write_header:
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    n_total = len(ground_truth) * len(configs)
    n_done = len(results)

    for entry in ground_truth:
        if entry['filename'] in already_done:
            continue

        fpath = os.path.join(dataset_dir, entry['filename'])
        this_gulp_size = gulp_size if gulp_size is not None else entry['fchans']

        for config_name, config in configs.items():
            try:
                result = find_et(fpath, config, device=device, gulp_size=this_gulp_size)
                hit_table = result.hit_table
            except Exception as e:
                logger.error(f"run_ablation: find_et failed on {entry['filename']} "
                             f"config={config_name}: {type(e).__name__}: {e}")
                row = {
                    'filename': entry['filename'],
                    'config': config_name,
                    'rfi_variant': entry['rfi_variant'],
                    'snr_requested': entry['et_snr_requested'],
                    'snr_measured_local': entry['et_snr_measured_local'],
                    'drift_rate_hz_s': entry['et_drift_rate_hz_s'],
                    'recovered': False,
                    'measured_snr': None,
                    'n_false_positives': None,
                    'error': str(e),
                }
                results.append(row)
                pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False)
                n_done += 1
                if n_done % 50 == 0:
                    logger.info(f"run_ablation: progress {n_done}/{n_total}")
                continue

            recovered, measured_snr, n_fp = match_hit_to_truth(
                hit_table, entry['et_f_start_chan'], entry['et_drift_rate_hz_s'],
            )

            row = {
                'filename': entry['filename'],
                'config': config_name,
                'rfi_variant': entry['rfi_variant'],
                'snr_requested': entry['et_snr_requested'],
                'snr_measured_local': entry['et_snr_measured_local'],
                'drift_rate_hz_s': entry['et_drift_rate_hz_s'],
                'recovered': recovered,
                'measured_snr': measured_snr,
                'n_false_positives': n_fp,
                'error': None,
            }
            results.append(row)
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False)
            n_done += 1
            if n_done % 50 == 0:
                logger.info(f"run_ablation: progress {n_done}/{n_total}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    logger.info(f"run_ablation: wrote {len(results_df)} rows to {output_csv}")
    return results_df


def compute_precision_f1(results_df):
    """ Add precision and F1 columns to a per-(config[,snr_requested])
    summary table produced by summarize() or an equivalent groupby.

    Precision here is computed as total_TP / (total_TP + total_FP)
    aggregated across trials in each group, NOT as a per-trial average of
    per-trial precision -- the latter is undefined whenever a trial has
    TP=0 and FP=0 (no signal recovered, no false positive either), and
    silently averaging over those undefined cases would distort the
    result. Aggregating TP and FP first, then dividing once, avoids that.
    """
    df = results_df.copy()
    agg = df.groupby('config').agg(
        recall=('recovered', 'mean'),
        total_TP=('recovered', 'sum'),
        total_FP=('n_false_positives', 'sum'),
        n_trials=('recovered', 'count'),
    ).reset_index()
    agg['precision'] = (agg['total_TP'] / (agg['total_TP'] + agg['total_FP'])).fillna(0)
    agg['f1'] = (2 * agg['precision'] * agg['recall'] / (agg['precision'] + agg['recall'])).fillna(0)
    return agg


def compute_precision_f1_by_snr(results_df, configs_to_compare):
    """ Same as compute_precision_f1, but grouped by (config,
    snr_requested) -- this is what reveals the SNR-dependent crossover
    (KLT preferable above some SNR threshold, raw preferable below it)
    that the aggregate-only table hides. See module docstring / paper
    methods section for why this disaggregation matters: averaging over
    all SNR levels masked a real threshold effect in initial testing.
    """
    df = results_df[results_df['config'].isin(configs_to_compare)].copy()
    agg = df.groupby(['config', 'snr_requested']).agg(
        recall=('recovered', 'mean'),
        total_TP=('recovered', 'sum'),
        total_FP=('n_false_positives', 'sum'),
        n_trials=('recovered', 'count'),
    ).reset_index()
    agg['precision'] = (agg['total_TP'] / (agg['total_TP'] + agg['total_FP'])).fillna(0)
    agg['f1'] = (2 * agg['precision'] * agg['recall'] / (agg['precision'] + agg['recall'])).fillna(0)
    return agg


def plot_metrics_vs_snr(results_df, configs_to_compare, labels, output_path):
    """ Recall / precision / F1 vs SNR, for a small set of configs
    (typically 'raw' vs the best-performing KLT config found by
    compute_precision_f1). Categorical x-axis (not log/linear-continuous)
    since the SNR grid is sparse (4 points by default) and a continuous
    axis visually implies a resolution the data doesn't have.

    Args:
        results_df (pd.DataFrame): output of run_ablation()
        configs_to_compare (list[str]): config names to plot (2-4
                  recommended; more becomes visually crowded)
        labels (dict): {config_name: display_label}
        output_path (str): where to save the .png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = compute_precision_f1_by_snr(results_df, configs_to_compare)
    snr_levels = sorted(summary['snr_requested'].unique())
    x_pos = range(len(snr_levels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ['recall', 'precision', 'f1']
    titles = ['Recall', 'Precision', 'F1 Score']
    markers = ['o', 's', '^', 'D']

    for ax, metric, title in zip(axes, metrics, titles):
        for i, config in enumerate(configs_to_compare):
            sub = summary[summary['config'] == config].sort_values('snr_requested')
            ax.plot(x_pos, sub[metric].values, marker=markers[i % len(markers)],
                     linestyle='-', linewidth=2, markersize=9, label=labels.get(config, config))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(snr_levels)
        ax.set_xlabel('Requested SNR (setigen nominal)')
        ax.set_ylabel(title)
        ax.set_title(title + ' vs SNR')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"plot_metrics_vs_snr: saved {output_path}")


def plot_recall_heatmap_2d(results_df, output_path, metric='recovered', metric_label='Recall'):
    """ Heatmap of a metric (default: recall) over the (var_frac,
    klt_window) grid, averaged across all SNR/drift levels. Excludes the
    'raw' config (it has no var_frac/klt_window to pivot on). Uses
    categorical tick labels rather than placing rows/columns at their
    literal numeric positions, since var_frac values are not evenly
    spaced (0.3, 0.5, 0.7, 0.95) and imshow would otherwise visually
    imply even spacing.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = results_df[results_df['config'] != 'raw'].copy()
    extracted = df['config'].str.extract(r'klt_vf([\d.]+)_w(\d+)')
    df['var_frac'] = extracted[0].astype(float)
    df['klt_window'] = extracted[1].astype(int)

    pivot = df.groupby(['var_frac', 'klt_window'])[metric].mean().unstack('klt_window')

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(0.1, pivot.values.max())
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index])
    ax.set_xlabel('KLT window (channels)')
    ax.set_ylabel('var_frac (cumulative variance threshold)')
    ax.set_title(f'{metric_label}, averaged over all SNR/drift levels')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                     color='white' if val < 0.6 * vmax else 'black')
    plt.colorbar(im, ax=ax, label=metric_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"plot_recall_heatmap_2d: saved {output_path}")


def plot_false_positives_total(results_df, output_path):
    """ Bar chart of total false positives per config, across the whole
    dataset. This is the single most visually direct result of the
    ablation (raw accumulates hundreds of false positives from
    incompletely-removed stationary RFI; every KLT configuration tested
    drives this to exactly zero on this dataset) -- worth its own figure
    rather than only appearing as a column in a table.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fp_total = results_df.groupby('config')['n_false_positives'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['tab:red' if c == 'raw' else 'tab:green' for c in fp_total.index]
    ax.bar(range(len(fp_total)), fp_total.values, color=colors)
    ax.set_xticks(range(len(fp_total)))
    ax.set_xticklabels(fp_total.index, rotation=45, ha='right')
    ax.set_ylabel(f'Total false positives (across {results_df["filename"].nunique()} files)')
    ax.set_title('False positive count: raw vs every KLT configuration tested')
    for i, v in enumerate(fp_total.values):
        ax.text(i, v + max(fp_total.values) * 0.02, str(int(v)), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"plot_false_positives_total: saved {output_path}")


def generate_paper_figures(results_df, output_dir, best_klt_config=None):
    """ Convenience wrapper: generates all three standard figures into
    output_dir. If best_klt_config is None, picks the config with the
    highest aggregate F1 score (excluding 'raw') automatically.
    """
    os.makedirs(output_dir, exist_ok=True)

    if best_klt_config is None:
        agg = compute_precision_f1(results_df)
        agg_klt = agg[agg['config'] != 'raw']
        best_klt_config = agg_klt.loc[agg_klt['f1'].idxmax(), 'config']
        logger.info(f"generate_paper_figures: auto-selected best KLT config by F1: {best_klt_config}")

    plot_metrics_vs_snr(
        results_df, ['raw', best_klt_config],
        labels={'raw': 'Raw (no KLT)', best_klt_config: f'KLT ({best_klt_config})'},
        output_path=os.path.join(output_dir, 'fig1_metrics_vs_snr.png'),
    )
    plot_recall_heatmap_2d(
        results_df, output_path=os.path.join(output_dir, 'fig2_recall_heatmap.png'),
    )
    plot_false_positives_total(
        results_df, output_path=os.path.join(output_dir, 'fig3_false_positives_total.png'),
    )
    """ Recall (fraction recovered) and mean false-positive count per
    (config, snr_requested), averaged over drift rates and RFI variants
    -- the core table/plot for the paper's results section.
    """
    summary = results_df.groupby(['config', 'snr_requested']).agg(
        recall=('recovered', 'mean'),
        mean_false_positives=('n_false_positives', 'mean'),
        n_trials=('recovered', 'count'),
    ).reset_index()
    return summary


def summarize_2d(results_df, snr_filter=None):
    """ Pivot recall into a (var_frac x klt_window) grid, for a fixed
    snr_requested (or averaged across all SNRs if snr_filter is None).
    Only meaningful when run_ablation was called with klt_window_scan
    set (otherwise every klt config shares window=256 and this collapses
    to a single column).

    Args:
        results_df (pd.DataFrame): output of run_ablation()
        snr_filter (float or None): restrict to one SNR level, or
                  average across all if None

    Returns:
        pivot (pd.DataFrame): rows=var_frac, columns=klt_window,
                  values=recall. The 'raw' config is excluded (it has no
                  var_frac/klt_window to pivot on).
    """
    df = results_df[results_df['config'] != 'raw'].copy()
    if snr_filter is not None:
        df = df[df['snr_requested'] == snr_filter]

    # Parse 'klt_vf{vf}_w{kw}' back into numeric columns.
    extracted = df['config'].str.extract(r'klt_vf([\d.]+)_w(\d+)')
    df['var_frac'] = extracted[0].astype(float)
    df['klt_window'] = extracted[1].astype(int)

    pivot = df.groupby(['var_frac', 'klt_window'])['recovered'].mean().unstack('klt_window')
    return pivot


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run RFI-removal / dedoppler ablation")
    parser.add_argument('-d', '--dataset-dir', required=True, help="Directory with .h5 files + ground_truth.json")
    parser.add_argument('-o', '--output-csv', required=True, help="Path to write results CSV")
    parser.add_argument('--max-dd', type=float, default=8.0)
    parser.add_argument('--min-fdistance', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=8.0)
    parser.add_argument('--var-frac-scan', type=float, nargs='+', default=[0.95, 0.7, 0.5, 0.3])
    parser.add_argument('--klt-window-scan', type=int, nargs='+', default=None,
                         help="If given, scans klt_window x var_frac (2D grid). If omitted, klt_window fixed at 256.")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--figures-dir', default=None,
                         help="If given, generates the standard paper figures (metrics vs SNR, "
                              "2D recall heatmap, false-positive bar chart) into this directory.")
    args = parser.parse_args()

    df = run_ablation(
        args.dataset_dir, args.output_csv,
        max_dd=args.max_dd, min_fdistance=args.min_fdistance,
        threshold=args.threshold, var_frac_scan=tuple(args.var_frac_scan),
        klt_window_scan=tuple(args.klt_window_scan) if args.klt_window_scan else None,
        device=args.device,
    )
    print(summarize(df).to_string(index=False))
    if args.klt_window_scan:
        print()
        print("Recall, var_frac x klt_window (averaged over all SNR levels):")
        print(summarize_2d(df).to_string())

    if args.figures_dir:
        generate_paper_figures(df, args.figures_dir)
        print()
        print(f"Figures written to {args.figures_dir}")
