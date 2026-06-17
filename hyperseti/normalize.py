import numpy as np
import time
import os

from .xp_compat import get_xp
from .data_array import DataArray

#logging
from .log import get_logger
logger = get_logger('hyperseti.normalize')


def normalize(data_array: DataArray,  mask=None, poly_fit: int=0):
    """ Apply normalization on GPU or CPU
    
    Applies normalisation (data - mean) / stdev
    
    Args: 
        data (DataArray): Data to preprocess (time, beam_id, frequency)
        mask (np.ndarray or cp.ndarray): 1D Channel mask for RFI flagging
        poly_fit (int): Fit polynomial of degree N, 0 = no fit.
        
    Returns: data_array (DataArray): Normalized data. The array module used
        (numpy or cupy) is whichever data_array.data already used coming in
        -- this function does not move data between devices.
    """
    # Dispatch on whatever array module produced this data; this is what
    # lets the same function body run unmodified on CPU and GPU.
    xp = get_xp(data_array.data)

    # Normalise
    logger.debug(f"Poly fit = {poly_fit}")
    t0 = time.time()
    
    # Get rid of NaNs - TODO: figure out why there are NaNs ...
    data_array.data = xp.nan_to_num(data_array.data)
    
    d_flag = xp.copy(data_array.data)

    n_int, n_ifs, n_chan = data_array.data.shape

    # Setup 1D channel mask -- used for polynomial fitting
    if mask is None: 
        mask = xp.zeros(n_chan, dtype='bool')

    # Do polynomial fit and compute stats (with masking)
    d_mean_ifs, d_std_ifs = xp.zeros(n_ifs), xp.zeros(n_ifs)
    if poly_fit > 0:
        d_poly_ifs = xp.zeros((n_ifs, poly_fit + 1))

    N_masked = mask.sum()
    N_flagged = N_masked * n_ifs * n_int
    N_tot     = np.prod(data_array.data.shape)
    N_unflagged = (N_tot - N_flagged)

    flag_fraction =  N_flagged / N_tot
    ## flag_correction =  N_tot / (N_tot - N_flagged) # <---------------------- unused
    logger.debug(f"Flagged fraction: {flag_fraction:2.4f}")
    if flag_fraction > 0.2:
        logger.warning(f"High flagged fraction: {flag_fraction:2.3f}")
    if flag_fraction > 0.98:
        logger.critical(f"Too much data flagged: {flag_fraction:2.3f}")
        
        # Ignore mask and ignore this data channel
        # TODO: How to make user notice if in a batch run?
        mask = xp.zeros(n_chan, dtype='bool')
        data_array.data = xp.ones_like(data_array.data)

    t0p = time.time()
    
    for ii in range(n_ifs):
        x    = xp.arange(n_chan, dtype='float64') 
        xc   = xp.compress(~mask, x)
        dfit = xp.compress(~mask, data_array.data[:, ii].mean(axis=0))

        if poly_fit > 0:
            try:
                # WAR: int64 dtype causes issues in cupy 10 (19.04.2022)
                poly_coeffs = xp.polyfit(xc, dfit, poly_fit)
                p    = xp.poly1d(poly_coeffs)
                fit   = p(x)
                dfit  -=  p(xc)
                data_array.data[:, ii] = data_array.data[:, ii] - fit
            except TypeError:
                # WAR for TypeError: expected non-empty vector for x 
                logger.critical(f"Error encountered in poly fitting!")
                poly_coeffs = np.zeros(poly_fit)
                dfit = xp.compress(~mask, data_array.data[:, ii].mean(axis=0))
            
            d_poly_ifs[ii] = poly_coeffs
        

        # compute mean and stdev
        dmean = xp.nanmean(dfit)
        dvar  = xp.nanmean((data_array.data[:, ii] - dmean)**2, axis=0)
        dvar  = xp.nanmean(xp.compress(~mask, dvar))
        dstd  = xp.sqrt(dvar)
        d_mean_ifs[ii] = dmean
        d_std_ifs[ii]  = dstd

    t1p = time.time()
    logger.debug(f"Mean+Std time: {(t1p-t0p)*1e3:2.2f}ms")


    # Add means and STDEV as attributes to data array
    pp_dict = { 'mean': d_mean_ifs, 
                'std': d_std_ifs,
                'flagged_fraction': flag_fraction,
                }
                
    if poly_fit > 0:
        pp_dict['n_poly'] = poly_fit
        pp_dict['poly_coeffs'] = d_poly_ifs

    # Attach preprocessing data to data array
    data_array.attrs['preprocess'] = pp_dict

    #  Apply to original data
    for ii in range(n_ifs):
        data_array.data[:, ii] = ((data_array.data[:, ii] - d_mean_ifs[ii]) / d_std_ifs[ii])
    
    t1 = time.time()
    logger.debug(f"Normalisation time: {(t1-t0)*1e3:2.2f}ms")
    
    return data_array
