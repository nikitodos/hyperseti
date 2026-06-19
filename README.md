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

