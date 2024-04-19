import numpy as np
import matplotlib.pyplot as plt
import os, sys
from mpi4py import MPI

from nbodykit.lab import BigFileCatalog
# import illustris_python as il

root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

basePathFastPM_seed = '/global/cfs/cdirs/cmb/data/halfdome/stampede2_3750Mpch_6144cube/final_res/3750Mpc_2048node_seed_%d/rfof_proc131072_nc6144_size3750_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s%d_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8'
seeds_fastpm = np.arange(100,122,2)


redshifts = [0, 1, 2, 3]
V_fastpm = (3.75e3)**3

# load catalog once to get some global params (M0)
rfof = BigFileCatalog(basePathFastPM_seed % (100,100) + '/rfof_%.4f' % 1, dataset='RFOF', comm=comm)
delta_mass = rfof.attrs['M0']
mass_min = delta_mass * 22
    
bin_width = 0.5 # np.log10(delta_mass*10)
bin_edges = np.arange(np.log10(mass_min), np.log10(mass_min)+4, bin_width)
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

if rank == root:
    filearr = np.empty((len(bin_centers), 1+len(redshifts)))

for zi,z in enumerate(redshifts):
    if rank == root: print('z',z)
    a = 1/(1+z)
    
    # load fastpm halos
    N_fastpm_seeds = np.zeros((len(seeds_fastpm), len(bin_centers)), dtype=np.int64)
    for si,seed in enumerate(seeds_fastpm):
        if rank == root: print('seed',seed)
        sys.stdout.flush()
        rfof = BigFileCatalog(basePathFastPM_seed % (seed,seed) + '/rfof_%.4f' % a, dataset='RFOF', comm=comm)
        M_fastpm = rfof['Length'].compute() * rfof.attrs['M0']
        N_fastpm, _ = np.histogram(np.log10(M_fastpm), bins=bin_edges, density=False)
        comm.Allreduce(MPI.IN_PLACE, N_fastpm, op=MPI.SUM)     # gather and add all histograms from all procs
        N_fastpm_seeds[si] = N_fastpm
        
    if rank == root:
        # compute avg and std of fsatpm halos over all seeds
        N_fastpm = np.mean(N_fastpm_seeds, axis=0)
        errN_fastpm = np.std(N_fastpm_seeds, axis=0)
        
        # compute ratio and make plot
        n_fastpm = N_fastpm / V_fastpm
        
        R = n_fastpm
        # errR = errN_fastpm / V_fastpm / n_tng
        
        plt.errorbar(bin_centers+10, R, yerr=errR, label=r'$z=%d$'%z)
        
        # save to file
        if zi == 0:
            filearr[:,0] = bin_centers+10
        filearr[:,zi+1] = R
        
if rank == root:
    np.savetxt('HMF.txt', filearr, header='R = HMF_fastpm\nlog10M[Msun/h] ' + ''.join(['R(z=%.1f) '%z for z in redshifts]))
