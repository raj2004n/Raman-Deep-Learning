import ramanspy as rp
import matplotlib.pyplot as plt
from src.analysis.endmember_estimator import estimate_endmembers

def show_unmixing_viewer(hsi_cube, n_endmembers, start=None, end=None):
    # crop
    if start is not None or end is not None:
        cropper = rp.preprocessing.misc.Cropper(region=(start, end))
        hsi_cube = cropper.apply(hsi_cube)

    # estimate number of endmembers if requested
    if n_endmembers == -1:
        n_endmembers, confidence = estimate_endmembers(hsi_cube)
        print(f"Estimated {n_endmembers} endmembers with {confidence} confidence.")
    
    nfindr = rp.analysis.unmix.NFINDR(n_endmembers=n_endmembers, abundance_method='fcls')
    abundance_maps, endmembers = nfindr.apply(hsi_cube)

    # plot endmember spectra
    rp.plot.spectra(
        endmembers, hsi_cube.spectral_axis,
        plot_type="single stacked",
        label=[f"Endmember {i + 1}" for i in range(len(endmembers))]
    )
    plt.show()

    # plot abundance maps
    rp.plot.image(
        abundance_maps,
        title=[f"Component {i + 1}" for i in range(len(abundance_maps))]
    )
    plt.show()