import math
import numpy as np
import ramanspy as rp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from .theme import apply_theme, BG, FG, GRID, ACCENT
from src.analysis.endmember_estimator import estimate_endmembers

def show_unmixing_viewer(hsi_cube, n_endmembers, start=None, end=None):
    apply_theme()   
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
    
    ax = rp.plot.spectra(
        endmembers, hsi_cube.spectral_axis,
        plot_type="single stacked",
        label=[f"Endmember {i + 1}" for i in range(len(endmembers))]
    )

    magma = cm.get_cmap("magma")
    lines = ax.get_lines()
    colors = magma(np.linspace(0.15, 0.85, len(lines)))

    for line, color in zip(lines, colors):
        line.set_color(color)

    plt.show()
    n = len(abundance_maps)
    cmap="magma"
    if n <= 10:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for i, (abundance_map, ax) in enumerate(zip(abundance_maps, axes)):
            rp.plot.image(
                abundance_map,
                title=f"Component {i + 1}",
                cmap=cmap,
                ax=ax
            )

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    else:
        # too many — show individually
        for i, abundance_map in enumerate(abundance_maps):
            rp.plot.image(
                abundance_map,
                title=f"Component {i + 1}",
                cmap=cmap
            )
            plt.show()