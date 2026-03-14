"""
Space to test pysptools. Focusign on Hysime
"""
from notebooks.raman_helper import *
from pysptools import material_count
# load data
path = Path("~/Code/Data_SH/SB008").expanduser()
raman_data = Raman_Data(path, 10, 13)

# raman slice containing 
all_slices = raman_data.get_raw_hsi_cube()

preprocessing_pipeline = rp.preprocessing.Pipeline([
    rp.preprocessing.misc.Cropper(region=(200, 1200))
])

#all_slices = preprocessing_pipeline.apply(all_slices)

# get the spectral data and axis
M = all_slices.spectral_data
spectral_data = all_slices.spectral_data
spectral_axis = all_slices.spectral_axis

hfc = material_count.HfcVd()
M_scaled = (spectral_data - spectral_data.min()) / (spectral_data.max() - spectral_data.min())
vd = hfc.count(M, noise_whitening=True)
print(f"Signal Subspace dimension: {vd}")


"""
nfindr = rp.analysis.unmix.NFINDR(n_endmembers=hysime.kf, abundance_method='fcls')
abundance_maps, endmembers = nfindr.apply(all_slices)

rp.plot.spectra(
    endmembers, all_slices.spectral_axis, 
    plot_type="single stacked", 
    label=[f"Endmember {i + 1}" for i in range(len(endmembers))]
    )
plt.show()

rp.plot.image(abundance_maps,
              title=[f"Component {i + 1}" for i in range(len(abundance_maps))]
              )

plt.show()"""