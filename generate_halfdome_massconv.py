# python generate_halfdome_massconv.py

import hmf  # for WatsonFoF
import hmf_emulator as hmfe  # for Aemulus

import numpy as np
from scipy.interpolate import CubicSpline
from astropy.cosmology import FlatLambdaCDM


h = 0.6774
OmegaB = 0.0486
OmegaM = 0.3089

# set up HMF cosmologies
aemulus_emu = hmfe.hmf_emulator()
cosmology={
    "omega_b": OmegaB * h**2,
    "omega_cdm": (OmegaM - 0.0486)  * h**2,
    "w0": -1.0,
    "n_s": 0.9667,
    "ln10As": 3.04478383,
    "H0": h * 100,
    "N_eff": 3.044
}
aemulus_emu.set_cosmology(cosmology)
flat_lcdm = FlatLambdaCDM(H0 = cosmology['H0'], Om0=OmegaM, Tcmb0=2.725, Ob0=OmegaB)


Mmin, Mmax, Mstep = 10.0, 16.0, 0.01
log10_m = np.arange(Mmin, Mmax, Mstep)
m_grid = 10**log10_m
z_grid = np.linspace(0,4,21)
watson_transform_grid =  np.zeros( (len(z_grid), len(log10_m)) )
aemulus_transform_grid =  np.zeros( (len(z_grid), len(log10_m)) )

lm_pivot = 12.5
def sigmoid(x,mi, mx): return mi + (mx-mi)*(lambda t: (1+200**(-t+0.5))**(-1) )( (x-mi)/(mx-mi) )
def step_amp(z): return -0.0125 * z**2 + 0.0945 * z + 0.0195

for zi, z in enumerate(z_grid):
    mf = hmf.MassFunction(Mmin=Mmin, Mmax=Mmax, dlog10m=Mstep, z=z, 
                          hmf_model="Watson_FoF", cosmo_model=flat_lcdm)
    step_m = 1 - step_amp(z) * (-1 + sigmoid(log10_m - lm_pivot, 0, 1))
    
    # cumulative integral backwards
    ngtm_1 = hmf.mass_function.integrate_hmf.hmf_integral_gtm(m_grid, mf.dndm)
    ngtm_2 = hmf.mass_function.integrate_hmf.hmf_integral_gtm(m_grid, mf.dndm * step_m)
    C_1_inv = CubicSpline( np.log10(ngtm_1[::-1]), log10_m[::-1])
    C_2 = CubicSpline(log10_m, np.log10(ngtm_2) )
    new_m = C_1_inv(C_2(log10_m))
    watson_transform_grid[zi,:] = new_m
    
    aemulus_mf = aemulus_emu.dndM(m_grid, z)
    ngtm_1 = hmf.mass_function.integrate_hmf.hmf_integral_gtm(m_grid, aemulus_mf)
    C_1_inv = CubicSpline( np.log10(ngtm_1[::-1]), log10_m[::-1])
    new_m = C_1_inv(C_2(log10_m))
    aemulus_transform_grid[zi,:] = new_m

np.savetxt("data/watson_transformed_mass_grid.dat", watson_transform_grid)
np.savetxt("data/aemulus_transformed_mass_grid.dat", aemulus_transform_grid)
np.savetxt("data/z_vec.dat", z_grid)
np.savetxt("data/log10m_vec.dat", log10_m)

# TO USE:
# import scipy.interpolate 
# itp = scipy.interpolate.RectBivariateSpline(z_grid, log10_m, watson_transform_grid)
