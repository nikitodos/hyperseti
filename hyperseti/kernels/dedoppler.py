import numpy as np

from ..xp_compat import cp, get_xp, HAS_GPU
from .kernel_manager import KernelManager
from .dedoppler_cpu import (
    dedoppler_kernel_cpu,
    dedoppler_kurtosis_kernel_cpu,
    dedoppler_with_kurtosis_kernel_cpu,
)
from ..log import get_logger
from ..filter import apply_boxcar
from ..data_array import DataArray

logger = get_logger('hyperseti.dedoppler')

# The CUDA RawKernels can only be compiled when a real CUDA device is
# available -- compiling them eagerly on a CPU-only system would crash
# at import time, before any code even gets a chance to choose the CPU
# path. They are therefore behind this guard and stay None on CPU-only
# systems; DedopplerMan.execute() never touches them in that case.
if HAS_GPU:
    dedoppler_kernel = cp.RawKernel(r'''
    extern "C" __global__
        __global__ void dedopplerKernel
            (const float *data, float *dedopp, int *shift, int F, int T)
            /* Each thread computes a different dedoppler sum for a given channel
            
             F: N_frequency channels
             T: N_time steps
            
             *data: Data array, (T x F) shape
             *dedopp: Dedoppler summed data, (D x F) shape
             *shift: Array of doppler corrections of length D.
                     shift is total number of channels to shift at time T
            */
            {
            
            // Setup thread index
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int d   = blockIdx.y;   // Dedoppler trial ID
            const int D   = gridDim.y;   // Number of dedoppler trials
            // Index for output array
            const int dd_idx = d * F + tid;
            float dd_val = 0;
            
            int idx = 0;
            for (int t = 0; t < T; t++) {
                                // timestep    // dedoppler trial offset
                idx  = tid + (F * t)      + (shift[d] * t / T);
                if (idx < F * T && idx > 0) {
                    dd_val += data[idx];
                  }
                  // Divide through by sqrt(T) to keep S/N same as input
                  float Tf = (float)T;
                  dedopp[dd_idx] = dd_val / sqrt(Tf);
                }
            }
    ''', 'dedopplerKernel')


    dedoppler_kurtosis_kernel = cp.RawKernel(r'''
    extern "C" __global__
        __global__ void dedopplerKurtosisKernel
            (const float *data, float *dedopp, int *shift, int F, int T, int N)
            /* Each thread computes a different dedoppler SK for a given channel
            
             F: N_frequency channels
             T: N_time steps
             N: N_acc number of accumulations averaged within time step
             
             *data: Data array, (T x F) shape
             *dedopp: Dedoppler spectralkurtosis data, (D x F) shape
             *shift: Array of doppler corrections of length D.
                     shift is total number of channels to shift at time T
            
            Note: output needs to be scaled by N_acc, number of time accumulations
            */
            {
            
            // Setup thread index
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int d   = blockIdx.y;   // Dedoppler trial ID
            const int D   = gridDim.y;   // Number of dedoppler trials

            // Index for output array
            const int dd_idx = d * F + tid;
            float S1 = 0;
            float S2 = 0;
            
            int idx = 0;
            for (int t = 0; t < T; t++) {
                                // timestep    // dedoppler trial offset
                idx  = tid + (F * t)      + (shift[d] * t / T);
                if (idx < F * T && idx > 0) {
                    S1 += data[idx];
                    S2 += data[idx] * data[idx];
                  }
                  dedopp[dd_idx] = (N*T+1)/(T-1) * (T*(S2 / (S1*S1)) - 1);
                }
            }
    ''', 'dedopplerKurtosisKernel')

    dedoppler_with_kurtosis_kernel = cp.RawKernel(r'''
    extern "C" __global__
        __global__ void dedopplerWithKurtosisKernel
            (const float *data, float *dedopp, float *dedopp_sk, int *shift, int F, int T, int N)
            /* Each thread computes a different dedoppler sum and DDSK for a given channel
            
             F: N_frequency channels
             T: N_time steps
             N: N_acc number of accumulations averaged within time step
             
             *data: Data array, (T x F) shape
             *dedopp: Dedoppler summed data, (D x F) shape
             *dedopp_sk: Dedoppler spectral kurtosis data, (D x F) shape
             *shift: Array of doppler corrections of length D.
                     shift is total number of channels to shift at time T
            */
            {
            
            // Setup thread index
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int d   = blockIdx.y;   // Dedoppler trial ID
            const int D   = gridDim.y;   // Number of dedoppler trials

            // Index for output array
            const int dd_idx = d * F + tid;
            float S1 = 0;
            float S2 = 0;
            float Tf = (float)T;
            
            int idx = 0;
            for (int t = 0; t < T; t++) {
                                // timestep    // dedoppler trial offset
                idx  = tid + (F * t)      + (shift[d] * t / T);
                if (idx < F * T && idx > 0) {
                    S1 += data[idx];
                    S2 += data[idx] * data[idx];
                  }
                  dedopp[dd_idx] = S1 / sqrt(Tf);
                  dedopp_sk[dd_idx] = (N*T+1)/(T-1) * (T*(S2 / (S1*S1)) - 1);
                }
            }

    ''', 'dedopplerWithKurtosisKernel')
else:
    dedoppler_kernel = None
    dedoppler_kurtosis_kernel = None
    dedoppler_with_kurtosis_kernel = None


class DedopplerMan(KernelManager):
    """ Kernel manager for smearing correction """
    def __init__(self):
        super().__init__('DedopplerMan')
        self.N_time  = None
        self.N_chan  = None
        self.N_beam  = None
        self.N_dopp  = None
        self.kernel  = None
        self.device  = None
    
    def init(self, N_time: int, N_beam: int, N_chan: int, N_dopp: int, kernel='dedoppler', device='gpu'):
        """ Initialize (or reinitialize) kernel 
        
        Args:
            N_dopp (int): Number of dedoppler trials in input data
            N_chan (int): Number of frequency channels
            device (str): 'gpu' or 'cpu'. Determines whether workspace
                           arrays are allocated with cupy (and a CUDA
                           grid/block is computed) or with numpy (no
                           grid/block needed -- execute() loops in Python
                           and calls the numpy kernel equivalents instead).
        """
        from ..xp_compat import resolve_device
        device = resolve_device(device)

        reinit = False
        if N_time != self.N_time: reinit = True
        if N_chan != self.N_chan: reinit = True
        if N_beam != self.N_beam: reinit = True
        if N_dopp != self.N_dopp: reinit = True
        if kernel != self.kernel: reinit = True
        if device != self.device: reinit = True

        if reinit:
            logger.debug(f'DedopplerMan: Reinitializing (device={device})')
            self.N_time = N_time
            self.N_chan = N_chan
            self.N_beam  = N_beam
            self.N_dopp  = N_dopp

            self.kernel = kernel
            self.device = device

            xp = cp if device == 'gpu' else np
            dtype = xp.float32

            # Allocate memory for dedoppler data (GPU: cupy, CPU: numpy)
            self.workspace['dedopp'] = xp.zeros((N_dopp, N_beam, N_chan), dtype=dtype)
            if self.kernel == 'ddsk':
                self.workspace['dedopp_sk'] =  xp.zeros((N_dopp, N_beam, N_chan), dtype=dtype)

            if N_beam > 1:
                self.workspace['_dedopp'] = xp.zeros((N_dopp, N_chan), dtype=dtype)
                if self.kernel == 'ddsk':
                    self.workspace['_dedopp_sk'] =  xp.zeros((N_dopp, N_chan), dtype=dtype)
            else:
                self.workspace['_dedopp'] = self.workspace['dedopp']
                if self.kernel == 'ddsk':
                    self.workspace['_dedopp_sk'] =  self.workspace['dedopp_sk']

            # Setup grid and block dimensions (CUDA-specific; unused on CPU,
            # where execute() loops over drift trials in plain Python)
            if device == 'gpu':
                F_block = np.min((N_chan, 1024))
                F_grid  = N_chan // F_block
                self._grid = (F_grid, N_dopp)
                self._block = (F_block,)
            else:
                self._grid = None
                self._block = None


    def execute(self, data_array: DataArray, dd_shifts_gpu, boxcar_size: int=1) -> DataArray:
        """ Execute kernel to compute dedoppler array 
        
        Args:
            data_array (DataArray): input data
            dd_shifts_gpu (np.ndarray or cp.ndarray): per-trial channel
                shifts. Despite the legacy name, this can be a numpy array
                when self.device == 'cpu' (kept for backward compatibility
                with existing call sites).
            boxcar_size (int): boxcar filter size, see apply_boxcar
        """
        ws = self.workspace
        # Calculate number of integrations within each (time-averaged) channel
        samps_per_sec = np.abs((1.0 / data_array.frequency.step).to('s').value) / 2 # Nyq sample rate for channel
        N_acc = int(data_array.time.step.to('s').value / samps_per_sec)

        is_gpu = (self.device == 'gpu')
        # On CPU, dd_shifts may arrive as a plain numpy int array; the
        # per-kernel functions expect that directly, no conversion needed.

        # TODO: Candidate for parallelization
        for beam_id in range(self.N_beam):

            # Select out beam
            d_gpu = data_array.data[:, beam_id, :] 

            # Apply boxcar filter
            if boxcar_size > 1:
                d_gpu = apply_boxcar(d_gpu, boxcar_size=boxcar_size, mode='gaussian')
            
            if is_gpu:
                if self.kernel == 'dedoppler':
                    logger.debug(f"{type(d_gpu)}, {type(ws['dedopp'])}, {self.N_chan}, {self.N_time}")
                    dedoppler_kernel(self._grid, self._block, 
                                    (d_gpu, ws['_dedopp'], dd_shifts_gpu, self.N_chan, self.N_time)) # grid, block and arguments
                elif self.kernel == 'kurtosis':
                    # output must be scaled by N_acc, which can be figured out from df and dt metadata
                    logger.debug(f'dedoppler kurtosis: rescaling SK by {N_acc}')
                    logger.debug(f"dedoppler kurtosis: driftrates: {dd_shifts_gpu}")
                    dedoppler_kurtosis_kernel(self._grid, self._block,
                                    (d_gpu, ws['_dedopp'], dd_shifts_gpu, self.N_chan, self.N_time, N_acc)) # grid, block and arguments 
                elif self.kernel == 'ddsk':
                    # output must be scaled by N_acc, which can be figured out from df and dt metadata
                    logger.debug(f'dedoppler ddsk: rescaling SK by {N_acc}')
                    logger.debug(f"dedoppler ddsk: driftrates: {dd_shifts_gpu}")
                    dedoppler_with_kurtosis_kernel(self._grid, self._block,
                                    (d_gpu, ws['_dedopp'], ws['_dedopp_sk'], dd_shifts_gpu, self.N_chan, self.N_time, N_acc)) 
                else:
                    logger.critical("dedoppler: Unknown kernel={} !!".format(self.kernel))
                    raise RuntimeError("Dedoppler failed, unknown kernel!")
            else:
                # CPU path: numpy kernel equivalents return their result
                # rather than writing in-place (RawKernels write directly
                # into the buffer passed as an argument; plain numpy
                # functions don't need that convention), so we assign the
                # return value into the workspace slot ourselves here to
                # keep the rest of this method's bookkeeping unchanged.
                #
                # Shape note: when N_beam == 1, ws['_dedopp'] is an alias
                # for ws['dedopp'] with shape (N_dopp, 1, N_chan) (see
                # init()), but the CPU kernel functions return (N_dopp,
                # N_chan) since they only ever see a single 2D (T, F)
                # slice. reshape() to the destination's actual shape
                # rather than assuming a fixed rank here.
                shift_np = np.asarray(dd_shifts_gpu).astype(np.int32)
                if self.kernel == 'dedoppler':
                    result = dedoppler_kernel_cpu(d_gpu, shift_np, self.N_chan, self.N_time)
                    ws['_dedopp'][:] = result.reshape(ws['_dedopp'].shape)
                elif self.kernel == 'kurtosis':
                    logger.debug(f'dedoppler kurtosis (cpu): rescaling SK by {N_acc}')
                    result = dedoppler_kurtosis_kernel_cpu(d_gpu, shift_np, self.N_chan, self.N_time, N_acc)
                    ws['_dedopp'][:] = result.reshape(ws['_dedopp'].shape)
                elif self.kernel == 'ddsk':
                    logger.debug(f'dedoppler ddsk (cpu): rescaling SK by {N_acc}')
                    dd, dsk = dedoppler_with_kurtosis_kernel_cpu(d_gpu, shift_np, self.N_chan, self.N_time, N_acc)
                    ws['_dedopp'][:] = dd.reshape(ws['_dedopp'].shape)
                    ws['_dedopp_sk'][:] = dsk.reshape(ws['_dedopp_sk'].shape)
                else:
                    logger.critical("dedoppler: Unknown kernel={} !!".format(self.kernel))
                    raise RuntimeError("Dedoppler failed, unknown kernel!")
      
            if self.N_beam > 1:
                ws['dedopp'][:, beam_id] = ws['_dedopp']
                if self.kernel == 'ddsk':
                    ws['dedopp_sk'][:, beam_id] = ws['_dedopp_sk']

        if self.kernel == 'ddsk':
          return ws['dedopp'], ws['dedopp_sk']
        else:
          return ws['dedopp']