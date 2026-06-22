## hyperseti

<p align="right">
<a href="https://codecov.io/github/UCBerkeleySETI/hyperseti" > 
 <img src="https://codecov.io/github/UCBerkeleySETI/hyperseti/branch/master/graph/badge.svg?token=YGW53OTFQA"/> 
 </a>
<a href='https://hyperseti.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/hyperseti/badge/?version=latest' alt='Documentation Status' />
</a>
</p>

`Hyperseti` is a GPU-accelerated code for searching radio astronomy spectral datasets for 
narrowband technosignatures that indicate the presence of intelligent (i.e. technologically capable)
life beyond Earth. It was developed as part of the Breakthrough Listen initiative, which seeks to 
quantify the prevalence of intelligent life within the Universe.

Hyperseti is centered around a brute-force dedoppler CUDA code, and has a Python-based frontend.
Hyperseti is intended as the spiritual successor to the [turboSETI](https://github.com/UCBerkeleySETI/turbo_seti/)
package. 


#### Example Usage

The following code searches a filterbank file which contains telemetry data from the Voyager space mission 
([data available here](http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5)).


```python
from hyperseti.pipeline import find_et

voyager_h5 = '../test/test_data/Voyager1.single_coarse.fine_res.h5'

config = {
    'preprocess': {
        'sk_flag': True,                        # Apply spectral kurtosis flagging
        'normalize': True,                      # Normalize data
        'blank_edges': {'n_chan': 32},          # Blank edges channels
        'blank_extrema': {'threshold': 10000}   # Blank ridiculously bright signals before search
    },
    'dedoppler': {
        'kernel': 'ddsk',                       # Doppler + kurtosis doppler (ddsk)
        'max_dd': 10.0,                         # Maximum dedoppler delay, 10 Hz/s
        'apply_smearing_corr': True,            # Correct  for smearing within dedoppler kernel 
        'plan': 'stepped'                       # Dedoppler trial spacing plan (stepped = less memory)
    },
    'hitsearch': {
        'threshold': 20,                        # SNR threshold above which to consider a hit
    },
    'pipeline': {
        'merge_boxcar_trials': True             # Merge hits at same frequency that are found in multiple boxcars
    }
}

hit_browser = find_et(voyager_h5, config, gulp_size=2**20)
display(hit_browser.hit_table)

hit_browser.view_hit(0, padding=128, plot='dual')

```

Hyperseti can natively load data generated with [setigen](https://github.com/bbrzycki/setigen):

```python
import numpy as np
import cupy as cp
import pylab as plt
from astropy import units as u

import setigen as stg
from hyperseti.io import from_setigen
from hyperseti.dedoppler import dedoppler
from hyperseti.plotting import imshow_waterfall, imshow_dedopp

# Create data using setigen
frame = stg.Frame(...)

# Convert data into hyperseti DataArray
d = from_setigen(frame)
d.data = cp.asarray(d.data) # Copy to GPU

# Run dedoppler
dedopp_array = dedoppler(d, boxcar_size=1, max_dd=8.0, plan='optimal')

# Plot waterfall / dedoppler
plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
imshow_waterfall(d)
plt.subplot(1,2,2)
imshow_dedopp(dedopp_array)
plt.tight_layout()
```

![image](https://user-images.githubusercontent.com/713251/164058073-88ccf3b1-b4a1-4160-b650-fca37770f96d.png)

### Installation

> **Note (this fork):** the statement below ("a working CUDA environment is
> needed") applies to the upstream `hyperseti` package. This fork makes
> `cupy`/CUDA optional -- see [Fork changelog](#fork-changelog) below for
> what changed and how to install/run without a GPU.

Hyperseti uses the GPU heavily, so a working CUDA environment is needed, and
requires Python 3.7 or above.  hyperseti relies upon `cupy`, which is easiest to install using `conda` (or `mamba`). 

To install from [conda/mamba package](https://anaconda.org/technosignatures/hyperseti):

```
conda install -c technosignatures hyperseti
```

If starting from scratch, this should get you most of the way there:

```
conda create -n hyper -c nvidia -c conda-forge python=3.10 cupy jupyterlab ipywidgets
```

Jupyterlab and ipywidgets are optional, but useful for a base environment.

From there:

```
conda activate hyper
pip install git+https://github.com/ucberkeleyseti/hyperseti
```

---

## Fork changelog

This is `nikitodos/hyperseti`, branch `cpu-klt-dev`. It extends upstream
`hyperseti` with (1) a CPU/GPU-agnostic execution layer and (2) a windowed
Karhunen-Loève Transform (KLT) RFI-removal stage integrated into the
dedoppler preprocessing pipeline, developed for a submission to Acta
Astronautica (working title: *A Hybrid Karhunen-Loève / Brute-Force
Dedoppler Pipeline for Adaptive RFI Mitigation in Technosignature Searches*).

### Quick install (no GPU required)

```
pip install -e . --no-deps   # cupy is now an optional extra, not a hard dependency
pip install numpy scipy astropy h5py setigen blimpy pandas matplotlib
```

### [2026-06-17] CPU/GPU-agnostic core + KLT integration + synthetic validation

First snapshot delivered. Established the CPU-portable execution layer and
the core KLT integration, validated entirely on synthetic data.

**Added**
- `hyperseti/xp_compat.py`: array-dispatch module (`get_xp()`,
  `resolve_device()`, `to_device()`, `asnumpy()`). `cupy` becomes an
  optional, lazily-detected dependency instead of a hard import.
- `hyperseti/kernels/dedoppler_cpu.py`, `peak_finder_cpu.py`,
  `blank_hits_cpu.py`, `smear_corr_cpu.py`: numpy equivalents of the five
  CUDA `RawKernel`s, including explicit C-style integer truncation
  (`np.trunc`, not `//`) for negative trial drift rates, and explicit
  boundary masking matching the CUDA kernels' guard conditions.
- `hyperseti/klt.py`: windowed KLT module (vendored from `seti_klt`'s
  `klt/core.py`, ported to `xp_compat` dispatch). Provides `klt_denoise()`
  with independently toggleable RFI-removal and discarded-eigenvalue
  noise-estimation outputs.
- KLT preprocessing step wired into `GulpPipeline.preprocess()`, inserted
  after `normalize()` and before `blank_extrema()`.
- `validation/generate_test_set.py`: synthetic dataset generator (setigen)
  with known-ground-truth injected RFI and technosignature candidates.
- `validation/run_ablation.py`: ablation harness, `var_frac` x
  `klt_window` 2D scan, plotting functions, CSV results
  (`validation/figures/`).

**Changed**
- `pyproject.toml`: `cupy` moved from a hard dependency to an optional
  extra (`[tool.poetry.extras] gpu = ["cupy"]`).

**Key finding from this snapshot**
360-file synthetic ablation (30 RFI realizations x 4 SNR x 3 drift rates):
every KLT configuration eliminates false positives from synthetic
stationary RFI entirely (0 vs. 827 for raw dedoppler), but
`var_frac=0.95` (the original script's default) also eliminates the
genuine injected signal whenever it shares a window with the RFI. Best
identified configuration (`var_frac=0.30`, `klt_window=128`) more than
doubles F1 over raw dedoppler above an injected-SNR threshold of
~30-40, at the cost of zero recall below it.

**Known limitation of this snapshot, explicitly**: validated exclusively
on synthetic, single-character RFI (strong, stationary, single-channel)
constructed to be favorable to KLT-style subspace removal. This is the
gap the next entry addresses.

### [2026-06-18] Real-background injection-recovery harness + GPU cross-check script

Added the validation step needed to address the central weakness of the
previous snapshot: that "zero false positives" was measured exclusively
on synthetic RFI structurally favorable to the method. **Not yet run**
on this snapshot -- both scripts below are pipeline code, prepared ahead
of having the relevant data/hardware available, not yet executed against
real data or real GPU hardware. Treat their first real run as part of the
validation, not as a formality.

**Added**
- `validation/real_data/inspect_header.py`: read-only header inspector
  (`blimpy.Waterfall(..., load_data=False)`) for real filterbank files,
  to be run before any injection. Flags non-uniform parameters across a
  file set and non-singleton beam/IF axes explicitly, rather than
  assuming a single fixed file layout.
- `validation/real_data/inject_and_recover.py`: injection-recovery
  experiment on real filterbank backgrounds. Loads a real `.fil` file via
  `blimpy`, builds a `setigen.Frame` directly on top of it
  (`Frame.from_data(..., waterfall=wf)`, which lets setigen resolve the
  ascending/descending frequency convention from the file itself rather
  than guessing), calibrates the injected SNR/drift-rate grid from the
  file's own `tsamp`/`foff` rather than reusing the synthetic dataset's
  fixed Hz/s values, and runs raw-vs-KLT `find_et()` with the same
  hit-matching logic as the synthetic ablation. Designed against, but not
  yet run on, the FRB 121102 / Lovell Telescope dataset
  (Zenodo [10.5281/zenodo.3974768](https://zenodo.org/records/3974768));
  the dataset's actual RFI content has not been independently verified
  and should be characterized as part of running this script, not
  assumed. Extensive caveats on what this experiment can and cannot
  demonstrate are embedded directly in the script's module docstring.
- `benchmarks/cpu_gpu_crosscheck.py`: CPU/GPU numerical cross-check and
  throughput comparison, to be run on CUDA-capable hardware. Refuses to
  run (explicit early exit, not a silent CPU fallback) if
  `hyperseti.xp_compat.HAS_GPU` is `False`, so a benchmark accidentally
  run on the wrong machine fails loudly instead of producing a misleading
  CPU-only number labeled as a GPU result. Compares hit tables
  (channel/drift-rate, within search-grid-relative tolerance) between
  `device='cpu'` and `device='gpu'` on identical input, in both raw and
  KLT-enabled configurations.

**Restructured**
- `validation/` (previously developed and shipped from the separate
  `seti_klt` repository) is now part of this repository, so the full
  pipeline -- CPU/GPU core, KLT, and all validation/benchmark code -- is
  in one place. `seti_klt` (the original BSc thesis repository) is left
  unmodified.

**Verified in this snapshot (development-container-level checks only,
no GPU, no real data available in that environment)**:
- `blimpy.Waterfall.data` axis order `(time, beam, freq)` confirmed
  directly against `blimpy`'s `waterfall.py` source, not assumed by
  analogy.
- `setigen.Frame.save_fil()` -> `blimpy.Waterfall()` round-trip confirmed
  to produce a header `find_et()` can read.
- A `frame.df` unit-handling bug (bare float vs. expected `Quantity`) in
  `inject_and_recover.py`'s signal-width calculation, found and fixed
  before this snapshot.
- `cpu_gpu_crosscheck.py`'s no-GPU guard confirmed to exit before doing
  any work, on a machine without CUDA.

**Not yet done, explicitly**: `inject_and_recover.py` has not been run
against the real Zenodo dataset (not reachable from the development
container's network allowlist); `cpu_gpu_crosscheck.py` has not been run
on real GPU hardware. Both are the immediate next steps.

### [2026-06-20] Real-header calibration fix + three pre-existing single-trial dedoppler bugs found and fixed

Real headers from the FRB 121102 / Lovell archive were obtained (all 30
files: `nchans` 800 or 672 depending on session, `tsamp=256us`,
`foff=-0.5 MHz/chan`, `nifs=1`, 60.000s duration uniformly). Attempting
to actually run `inject_and_recover.py` against a file matching these
parameters (not yet the real archive itself -- still blocked by network
access, see below -- but a synthetic file built with identical header
parameters) surfaced four real bugs, none of which were visible from
code review alone.

**Fixed -- calibration bug (this repo's code, not upstream)**
- `calibrate_injection_grid()`'s drift-rate grid was expressed as a
  fraction of channel-width-per-timestep, reasoned (incorrectly) to
  "automatically scale" to any file's resolution. At this dataset's
  `tsamp=256us` (vs. the synthetic dataset's 18.25s), the same formula
  produced drift rates of 1-40 **million** Hz/s -- five to six orders
  of magnitude outside any physically realistic SETI range (~0.1-200
  Hz/s; Sheikh et al. 2019, arXiv:1910.01148). Fixed by using fixed,
  physically-motivated absolute Hz/s values instead, independent of the
  file's native resolution.
- Re-deriving the numbers properly then surfaced a second, more
  fundamental issue: this dataset's channel width (0.5 MHz, a backend
  built for broadband FRB detection, not narrowband SETI spectroscopy)
  is too coarse by ~6 orders of magnitude to resolve ANY physically
  realistic Doppler drift over the file's 60s duration (even 20 Hz/s
  displaces an injected signal by ~0.0024 channels). This is a property
  of the instrument, not a bug. `calibrate_injection_grid()`'s default
  was changed to a single near-zero drift value (0.01 Hz/s), and the
  real-data injection-recovery experiment is now explicitly scoped as a
  **stationary-signal detection / KLT-damage test on a real background**,
  not a dedoppler-search validation -- see the function's docstring for
  the full reasoning. A dedoppler-capable real-data test would need a
  fine-channelized dataset (a few Hz/channel or better), not this one.

**Found and fixed -- three pre-existing upstream bugs**, all
unreachable/unexercised until the change above actually required a
single-trial (`max_dd=0`) dedoppler search (confirmed via `git show
HEAD:<file>` against this fork's pre-CPU-port history that all three
predate this fork's changes):
- `hyperseti/dedoppler.py`: the `max_dd == 0 and min_dd is None`
  special case (meant to return a single drift=0 trial) was
  structurally unreachable -- `min_dd` is unconditionally overwritten
  from `None` to `-abs(max_dd)` a few lines before that check runs, so
  `min_dd is None` was always `False` by the time it was evaluated.
  Every single-trial request silently fell through to the general
  trial-grid construction instead, which raises `RuntimeError: No
  steps!` whenever the requested range rounds to zero channels (which
  it always does for `max_dd=0`, and also for the coarse-channel
  near-zero-drift case above). Fixed by recording whether `min_dd` was
  originally `None` before it gets overwritten.
- `hyperseti/kernels/peak_finder.py` (`PeakFinder.hitsearch`) and
  `hyperseti/hits.py` (`hitsearch`, in unused-but-latent code): both
  called a bare `.squeeze()` on a `(N_dopp, N_beam, N_chan)` array
  intending to drop only the size-1 beam axis. This is harmless when
  `N_dopp > 1` (only the beam axis is size-1), but with `N_dopp == 1`
  (the single-trial case the fix above enables) BOTH the drift and beam
  axes are size-1, and a bare `.squeeze()` collapses `(1, 1, N_chan)` to
  `(N_chan,)` instead of the required `(1, N_chan)`, failing a shape
  assertion downstream. Fixed by squeezing explicitly on the beam axis
  (`axis=1`) only, in both files.

**Verified**: the full chain (`inject_and_recover.py` end-to-end, raw
and KLT configurations, 4 SNR levels) now runs without error on a
synthetic file matching this dataset's real header parameters, with
`max_tchans` bounding the loaded time range (see below) rather than
loading the full ~234k-sample file every grid point.

**Added**
- `load_real_background()` / `run_one_file()` gained a `max_tchans`
  parameter (default 8192, CLI: `--max-tchans`). This dataset's files
  are ~234000 time samples each (60s at 256us) -- `find_et()`'s
  `gulp_size` bounds frequency channels per gulp, not time samples, so
  it does not bound this on its own; without `max_tchans`, every
  (file x SNR) grid point would reload and reprocess the full array.
- `inspect_header.py`: `nsamples` is absent from this archive's headers
  (and, it turns out, from any file `setigen.Frame.save_fil()` writes --
  confirmed empirically, not specific to this archive). Previously this
  silently skipped the file-duration calculation; now computed from
  the file's physical size on disk (same method `blimpy` uses
  internally, `blimpy.io.sigproc.calc_n_ints_in_file`), with the
  computed value clearly logged as computed rather than read.
- `inspect_all_filterbanks.bat`: a Windows batch script (a `.ps1`
  PowerShell equivalent was tried first but the user's environment
  rejected it -- not debugged further once the `.bat` worked) that
  loops `inspect_header.py` over every `.fil` file in a directory and
  writes a combined, timestamped log file alongside them.

**Not yet done, explicitly**: `inject_and_recover.py` still has not
been run against the actual archive files (only against a synthetic
file matching their header) -- the next immediate step, now that the
crashes blocking it are fixed.

