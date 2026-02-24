from raman_helper import *
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

path = Path("~/Code/Data_SH/SB008").expanduser()
raman_data = Raman_Data(path, 10, 13)

cell_layer = raman_data.get_raw_slices()

preprocessing_pipeline = rp.preprocessing.Pipeline([
    rp.preprocessing.misc.Cropper(region=(150, 1200))
])

preprocessed_cell_layer = preprocessing_pipeline.apply(cell_layer)

#nfindr = rp.analysis.unmix.NFINDR(n_endmembers=4, abundance_method='fcls')
#abundance_maps, endmembers = nfindr.apply(preprocessed_cell_layer)

fippi = rp.analysis.unmix.VCA(n_endmembers=3, abundance_method='nnls')
abundance_maps, endmembers = fippi.apply(preprocessed_cell_layer)

rp.plot.spectra(
    endmembers, preprocessed_cell_layer.spectral_axis, 
    plot_type="single stacked", 
    label=[f"Endmember {i + 1}" for i in range(len(endmembers))]
    )
plt.show()

rp.plot.image(abundance_maps,
              title=[f"Component {i + 1}" for i in range(len(abundance_maps))]
              )

plt.show()