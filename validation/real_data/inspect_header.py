"""
validation/real_data/inspect_header.py -- read-only inspection of the
real FRB 121102 / Lovell filterbank files (Zenodo 10.5281/zenodo.3974768),
to be run BEFORE any injection.

WHY THIS STEP EXISTS, EXPLICITLY:
This dataset's filterbank parameters were never directly verified by
us against the actual files -- the parameters assumed below (800
channels, 400 MHz bandwidth, 256 us sampling, 1532 MHz center
frequency) come from the published description of Lovell/Jodrell Bank
FRB 121102 observations in the same period as Rajwade et al. (2020),
NOT from reading these specific files. Different epochs/sessions in
this archive could plausibly differ. Running this script and reading
its printed output is a REQUIRED step, not an optional sanity check,
before choosing max_dd / klt_window for the injection experiment in
inject_and_recover.py -- those parameters are calibrated relative to
tsamp and nchans found here, not hardcoded to the assumed values above.

USAGE (run on the machine where the tarball was downloaded/extracted,
not in the development container this was written in -- see README in
this directory for why):
    python inspect_header.py /path/to/extracted/R1_Lovell_files/*.fil
"""
import sys
import glob

import numpy as np

try:
    import blimpy as bl
except ImportError:
    print("blimpy not installed. pip install blimpy --break-system-packages "
          "(or without that flag in a venv).")
    sys.exit(1)


def inspect_file(filepath):
    """ Read header only (load_data=False) -- safe even for the
    largest files in this archive, since we never load sample data here.
    """
    try:
        wf = bl.Waterfall(filepath, load_data=False)
    except Exception as e:
        print(f"  FAILED to read {filepath}: {type(e).__name__}: {e}")
        return None

    h = wf.header
    nchans = h.get('nchans')
    nsamples = h.get('nsamples')
    tsamp = h.get('tsamp')        # seconds
    fch1 = h.get('fch1')          # MHz, frequency of channel 0
    foff = h.get('foff')          # MHz, channel spacing (negative in
                                   # standard SIGPROC: frequency
                                   # decreases as channel index increases)
    nbits = h.get('nbits')
    nifs = h.get('nifs')

    duration_s = (nsamples * tsamp) if (nsamples and tsamp) else None
    bandwidth_mhz = (nchans * abs(foff)) if (nchans and foff) else None

    print(f"  nchans={nchans}  nsamples={nsamples}  nifs={nifs}  nbits={nbits}")
    if nifs is not None and nifs != 1:
        print(f"  WARNING: nifs={nifs} (not 1) -- inject_and_recover.py's "
              f"np.squeeze on the beam/IF axis assumes this axis has size "
              f"1 (single beam, single polarization product). A file with "
              f"nifs != 1 will produce a data array with more than 2 "
              f"dimensions after squeeze, which load_real_background() "
              f"explicitly checks for and raises on, rather than silently "
              f"misinterpreting the array shape.")
    print(f"  tsamp={tsamp*1e6:.2f} us" if tsamp else "  tsamp=MISSING")
    print(f"  fch1={fch1:.4f} MHz  foff={foff:.6f} MHz/chan" if (fch1 and foff) else "  fch1/foff=MISSING")
    if duration_s is not None:
        print(f"  -> file duration: {duration_s:.3f} s")
    if bandwidth_mhz is not None:
        print(f"  -> total bandwidth: {bandwidth_mhz:.2f} MHz")

    # Sign convention check: this matters because setigen (used for the
    # synthetic dataset elsewhere in this project) and SIGPROC have
    # opposite channel-frequency conventions in general use. Getting
    # this backwards would silently inject a signal with the WRONG sign
    # of drift rate relative to what the dedoppler grid expects.
    if foff is not None:
        sign = "decreasing" if foff < 0 else "increasing"
        print(f"  -> frequency vs. channel index is {sign} "
              f"(foff {'<' if foff < 0 else '>'} 0)")

    return dict(nchans=nchans, nsamples=nsamples, tsamp=tsamp,
                fch1=fch1, foff=foff, duration_s=duration_s)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepaths = []
    for pattern in sys.argv[1:]:
        filepaths.extend(sorted(glob.glob(pattern)))

    if not filepaths:
        print(f"No files matched: {sys.argv[1:]}")
        sys.exit(1)

    print(f"Inspecting {len(filepaths)} file(s)...\n")
    results = []
    for fp in filepaths:
        print(f"=== {fp} ===")
        r = inspect_file(fp)
        if r is not None:
            results.append(r)
        print()

    if len(results) > 1:
        # Flag inconsistency across files explicitly -- if these
        # observations span multiple sessions/epochs, parameters might
        # not be uniform, and the injection script assumes they are
        # (or needs per-file recalibration if not).
        nchans_set = set(r['nchans'] for r in results)
        tsamp_set = set(round(r['tsamp'], 9) if r['tsamp'] else None for r in results)
        print("--- Cross-file consistency check ---")
        print(f"Distinct nchans values across files: {nchans_set}")
        print(f"Distinct tsamp values across files: {tsamp_set}")
        if len(nchans_set) > 1 or len(tsamp_set) > 1:
            print("WARNING: parameters are NOT uniform across files. "
                  "The injection script's per-file calibration "
                  "(reading each file's own header) handles this "
                  "correctly, but any single fixed max_dd/klt_window "
                  "value will NOT be equally appropriate for every file.")


if __name__ == '__main__':
    main()
