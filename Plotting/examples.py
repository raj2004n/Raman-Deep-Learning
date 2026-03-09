import numpy as np
from raman_helper import *
import matplotlib.pyplot as plt
from pathlib import Path 
import numpy as np
import pandas as pd
import ramanspy as rp

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser() # store as Path object for easier manipulation
files = list(path.glob('*.txt')) # extract .txt files and store as list

if not files:
    print("No .txt files found in that directory.")

# read in spectra from one pixel
raman_shifts = pd.read_csv(
    files[0],
    sep='\t',
    names=['raman_shift'],
    header=None,
    usecols=[0]
    )['raman_shift'].tolist()

file = files[200] # only take the first file, as we would handle in a loop

intensity_arr = pd.read_csv(
        file, 
        sep='\t', 
        names=['intensity'], 
        header=None,
        usecols=[1],
        )['intensity'].tolist()

# denoise, not sure what i am lookng for here; how smooth given the spectra range
denoisers = [
    (rp.preprocessing.denoise.SavGol(window_length=10, polyorder=3), 'Savitzky-Golay'), 
    (rp.preprocessing.denoise.Whittaker(), 'Whittaker-Henderson'),
    (rp.preprocessing.denoise.Kernel(kernel_type='flat'), 'Kernel'),
    (rp.preprocessing.denoise.Gaussian(), 'Gaussian')
    ]

# baseline corrections
baseline_correctors = [
    # least squares
    (rp.preprocessing.baseline.ASLS(), 'AsLS'),
    (rp.preprocessing.baseline.IASLS(), 'IAsLS'),
    (rp.preprocessing.baseline.AIRPLS(), 'airPLS'),
    (rp.preprocessing.baseline.ARPLS(), 'arPLS'),
    (rp.preprocessing.baseline.DRPLS(), 'drPLS'),
    (rp.preprocessing.baseline.IARPLS(), 'IarPLS'),
    (rp.preprocessing.baseline.ASPLS(), 'asPLS'),
    ]

baseline_correctors2 = [
    # polynomial fitting
    (rp.preprocessing.baseline.Poly(poly_order=3), 'Polynomial'),
    (rp.preprocessing.baseline.ModPoly(poly_order=3), 'ModPoly'),
    (rp.preprocessing.baseline.PenalisedPoly(poly_order=3), 'PenalisedPoly'),
    (rp.preprocessing.baseline.IModPoly(poly_order=3), 'IModPoly')
    ]

baseline_correctors3 = [
    # other methods
    (rp.preprocessing.baseline.Goldindec(), 'Goldindec'),
    (rp.preprocessing.baseline.IRSQR(), 'IRSQR'),
    (rp.preprocessing.baseline.CornerCutting(), 'CornerCutting'),
    ]

normalisation = [
    (rp.preprocessing.normalise.Vector(), 'Vector'),
    (rp.preprocessing.normalise.MinMax(), 'MinMax'),
    (rp.preprocessing.normalise.MaxIntensity(), 'Max Intensity'),
    (rp.preprocessing.normalise.AUC(), 'AUC')
    ]

# spectra object
raw_spectra = rp.Spectrum(intensity_arr, raman_shifts)

# despike, relevant if using CCD according to this method
despike = rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=5)
ds_spectra = despike.apply(raw_spectra)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
axes_flat = axes.flatten()
fig.suptitle('Despike', fontsize=16)
rp.plot.spectra(raw_spectra, title='raw', ax=axes[0])
rp.plot.spectra(ds_spectra, title='despiked using WhitakerHayes (Uses Z-score)', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
axes_flat = axes.flatten()
fig.suptitle('Denoise', fontsize=16)
for i in range(len(denoisers)):
    rp.plot.spectra(denoisers[i][0].apply(ds_spectra), title=f"{denoisers[i][1]}", ax=axes_flat[i])
plt.show()

# despiked, denoised spectra
denoiser = rp.preprocessing.denoise.Whittaker()
ds_dn_spectra = denoiser.apply(ds_spectra)
fig, axes = plt.subplots(4, 2, figsize=(12, 24), tight_layout=True)
axes_flat = axes.flatten()
fig.suptitle('Baseline Correction using Least Square Fitting', fontsize=16)
# plot without baseline correction
rp.plot.spectra(ds_dn_spectra, title=f"Despiked and Denoised Spectra", ax=axes_flat[0])
j = 1
for i in range(len(baseline_correctors)):
    rp.plot.spectra(baseline_correctors[i][0].apply(ds_dn_spectra), title=f"{baseline_correctors[i][1]}", ax=axes_flat[j])
    j += 1
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 24), tight_layout=True)
axes_flat = axes.flatten()
fig.suptitle('Baseline Correction using Polynomial Fitting', fontsize=16)
# plot without baseline correction
rp.plot.spectra(ds_dn_spectra, title=f"Despiked and Denoised Spectra", ax=axes_flat[0])
j = 1
for i in range(len(baseline_correctors2)):
    rp.plot.spectra(baseline_correctors2[i][0].apply(ds_dn_spectra), title=f"{baseline_correctors2[i][1]}", ax=axes_flat[j])
    j += 1
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 24), tight_layout=True)
axes_flat = axes.flatten()
fig.suptitle('Baseline Correction using Other Fitting Methods', fontsize=16)
# plot without baseline correction
rp.plot.spectra(ds_dn_spectra, title=f"Despiked and Denoised Spectra", ax=axes_flat[0])
j = 1
for i in range(len(baseline_correctors3)):
    rp.plot.spectra(baseline_correctors3[i][0].apply(ds_dn_spectra), title=f"{baseline_correctors3[i][1]}", ax=axes_flat[j])
    j += 1

# handle fabc seperately because it doesn't like spectra object...
fabc = rp.preprocessing.baseline.FABC()
# get intensity and raman shifts from despiked and denoised spectra
intensity_arr, raman_shifts = ds_dn_spectra.spectral_data, ds_dn_spectra.spectral_axis
# applu fabc
fabc_arr, fabc_arr2 = fabc.method(intensity_arr, raman_shifts)
# create spectra object
fabc_spectra = rp.Spectrum(fabc_arr, fabc_arr2)
rp.plot.spectra(fabc_spectra, title='FABC', ax=axes_flat[j])
plt.show()

# despike, denoised, background corrected spectra
aspls = rp.preprocessing.baseline.ASPLS()
final_spectra = aspls.apply(ds_dn_spectra)

fig, axes = plt.subplots(3, 2, figsize=(12, 24), tight_layout=True)
axes_flat = axes.flatten()
fig.suptitle('Normalisation', fontsize=16)
# plot without normalisation
rp.plot.spectra(final_spectra, title=f"Despiked, Denoised, Background Corrected Spectra", ax=axes_flat[0])
j = 1
for i in range(len(normalisation)):
    rp.plot.spectra(normalisation[i][0].apply(final_spectra), title=f"{normalisation[i][1]}", ax=axes_flat[j])
    j += 1
plt.show()