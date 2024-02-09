"""
This script extracts the particle density grid of the initial conditions (ICs) from the 
Fourier IC file produced by FastPM. It depends critically on the input setting for FastPM
which specifies that the force mesh oversamples the particle IC mesh by a factor of 2.

**Example Usage**
```bash
INFILE='/global/cfs/cdirs/m3035/abayer/halfdome/ics/low_res/rfof_proc64_nc512_size5000_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_2048_rfofkdt_8/'
OUTFILE='delta3.hdf5'
srun -n 64 -c 2 python mesh2delta.py $INFILE $OUTFILE
```
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
from mpi4py import MPI
import h5py
from nbodykit.lab import *
from numba import jit

infile, outfile = sys.argv[1], sys.argv[2]  # args: the input mesh file and output h5 file

root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()  # no MPI vars are used directly, since layout is done by nbodykit
rank = comm.Get_rank()  # rank is useful for print-statement debugging

@jit(nopython=True)
def sublinearindices(A, rx, ry, rz, shape):
    '''Stores linear indices into A given sub-box index ranges rx, ry, rz. This function
    provides the linear index of each of the *local* IC particle locations.
    '''
    nx, nz, ny = shape
    for ai, i in enumerate(rx):
        for aj, j in enumerate(ry):
            for ak, k in enumerate(rz):
                A[ai,aj,ak] =  k + j*nz + i*ny*nz

mesh_k = BigFileMesh(infile +'lineark/', dataset='LinearDensityK')  # we store Fourier mesh
mesh_kr = mesh_k.compute(mode='real')  # big FFT to go to real density field

mesh_size = mesh_k.attrs['Nmesh']  # get size of global mesh, to know what particles to get
assert np.all(mesh_size % 2 == 0)  # make sure total mesh size is divisible by 2
particle_grid_size = tuple(map(lambda x: int(x/2), mesh_size))  # global particle mesh size 
n_particles = np.prod(particle_grid_size)  # total number of particles is size of hdf5 data 

region = mesh_kr.pm.domain.primary_region  # find region of the global mesh stored locally
i_starts_mesh = (region['start'][0]).astype(int)  # get start and ends of force mesh
i_ends_mesh = (region['end'][0]).astype(int)  # region is a float (?) so we convert to int
i_starts_pt = i_starts_mesh // 2   # local particle grid start index
i_ends_pt = i_ends_mesh // 2       # local particle grid end index
sub_sizes = i_ends_pt - i_starts_pt  # size of the local particle IC grid 

linear_inds = np.zeros(sub_sizes, dtype=int)  # linear indices of particle grid = ID
# global (i,j,k) indices along each axis of local particle grid
rx, ry, rz = [np.arange(i_starts_pt[i], i_ends_pt[i]) for i in range(3)]
sublinearindices(linear_inds, rx, ry, rz, particle_grid_size )  # get local IDs

deltas = mesh_kr[::2, ::2, ::2].flatten()  # will write flattened deltas, skip odd indices
linear_inds = linear_inds.flatten()        # we write those delta to linear index = ID

f = h5py.File(outfile, 'w', driver='mpio', comm=MPI.COMM_WORLD)  # MPI HDF5 writing
dset_delta = f.create_dataset('delta', (n_particles,), dtype='f4')  # Float32 deltas
dset_delta[linear_inds] = deltas  # each rank writes different linear_inds
f.close()
