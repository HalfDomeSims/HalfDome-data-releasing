import h5py
import gc, sys
import numpy as np
import scipy.interpolate 
from nbodykit.lab import BigFileCatalog

outdir = "/pscratch/sd/x/xzackli/hd/halos/"
chunk_size = 2**26
catdir = '/global/cfs/cdirs/cmb/data/halfdome/stampede2_3750Mpch_6144cube/final_res/3750Mpc_2048node_seed_%d/rfof_proc131072_nc6144_size3750_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s%d_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8'


seeds_fastpm = np.arange(100,122,2)
seed = seeds_fastpm[int(sys.argv[1])]  # easier to specify list of seeds in python
print("Converting seed ", seed)

def get_lightcone(filename, seed):
    catalog = BigFileCatalog(catdir % (seed,seed) + '/usmesh/',dataset='RFOF') #read the data
    log10_m = np.log10(catalog['Length'] * catalog.attrs['M0'] * 1e10)
    redshift = ((1 / catalog['Aemit']) - 1)
    n_halos = len(redshift)
    z_chunked = [(redshift[i:i + chunk_size]).compute() for i in range(0, n_halos, chunk_size)]
    lm_chunked = [(log10_m[i:i + chunk_size]).compute() for i in range(0, n_halos, chunk_size)]
    return lm_chunked, z_chunked, n_halos

# load in lightcone 
lm_chunked, z_chunked, n_halos = get_lightcone(catdir, seed)
gc.collect()

# set up interpolators
z_grid = np.genfromtxt("data/z_vec.dat")
log10_m = np.genfromtxt("data/log10m_vec.dat")
watson_grid = np.genfromtxt("data/watson_transformed_mass_grid.dat")
aemulus_grid = np.genfromtxt("data/aemulus_transformed_mass_grid.dat")

rfof_to_watson = scipy.interpolate.RectBivariateSpline(z_grid, log10_m, watson_grid)
rfof_to_aemulus = scipy.interpolate.RectBivariateSpline(z_grid, log10_m, aemulus_grid)

# write new masses
with h5py.File(f'{outdir}/transformed_masses_{seed}.hdf5', 'w') as f:
    dset_wat = f.create_dataset('watson', shape=n_halos, dtype='f4')
    dset_aem = f.create_dataset('aemulus', shape=n_halos, dtype='f4')
    
    for i, z_chunk, lm_chunk in zip(range(0, n_halos, chunk_size), z_chunked, lm_chunked):
        dset_wat[i:i + chunk_size] = 10**rfof_to_watson(z_chunk, lm_chunk, grid=False)
        dset_aem[i:i + chunk_size] = 10**rfof_to_aemulus(z_chunk, lm_chunk, grid=False)
        print(i, "/", n_halos)
