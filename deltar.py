import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

from nbodykit.lab import *

root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# 1/16 (1 node)
perl = '/global/cfs/cdirs/m3035/abayer/halfdome/ics/low_res/rfof_proc64_nc512_size5000_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_2048_rfofkdt_8/'
stamp = '/global/cfs/cdirs/cmb/data/halfdome/stampede2_3750Mpch_6144cube/lower_res/ode_16_res/rfof_proc64_nc512_size5000_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_2048_rfofkdt_8/'

# 1/8 (8 node)
perl  = '/global/cfs/cdirs/m3035/abayer/halfdome/ics/low_res/rfof_proc512_nc1024_size5000_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_2048_rfofkdt_8/'
stamp = '/global/cfs/cdirs/cmb/data/halfdome/stampede2_3750Mpch_6144cube/lower_res/ode_08_res/rfof_proc512_nc1024_size5000_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_2048_rfofkdt_8/'

# 1/4 (64 node)
perl = '/global/cfs/cdirs/m3035/abayer/halfdome/ics/low_res/rfof_proc4096_nc2048_size5000_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_2048_rfofkdt_8/'
stamp = '/global/cfs/cdirs/cmb/data/halfdome/stampede2_3750Mpch_6144cube/lower_res/ode_04_res/rfof_proc4096_nc2048_size5000_nsteps60lin_ldr0_rcvtrue_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8/'

print(perl)
print(stamp)

f = perl

a = 1
z = 1/a-1

mesh_r = BigFileMesh(f+'linearr/', dataset='LinearDensityR')
mesh_k = BigFileMesh(f+'lineark/', dataset='LinearDensityK')

t0 = time.time()
mesh_rr = mesh_r.compute(mode='real')
print(rank,'rr',time.time()-t0)
t0 = time.time()
mesh_rk = mesh_r.compute(mode='complex')
print(rank,'rk',time.time()-t0)
t0 = time.time()
mesh_kr = mesh_k.compute(mode='real')
print(rank,'kr',time.time()-t0)
t0 = time.time()
mesh_kk = mesh_k.compute(mode='complex')
print(rank,'kk',time.time()-t0)

print(rank,np.allclose(mesh_rr, mesh_kr, rtol=1e-6, atol=1e-5))

"""
i = 20
plt.imshow(mesh_rr[i])
plt.colorbar()
if rank == root: plt.show()
plt.imshow(mesh_kr[i])
plt.colorbar()
if rank == root: plt.show()
plt.imshow(mesh_rr[i]/mesh_kr[i])
plt.colorbar()
if rank == root: plt.show()
"""

print(rank,np.allclose(mesh_rk, mesh_kk, rtol=1e-6, atol=1e-6))
