import argparse
from pathlib import Path

import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RangeSlider
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from data.raman_helper import Raman_Data
from data_analysis import Analysis
"""
e.g. input:

python3 plot_unmixing.py ~/Code/Data_SH/SB008 10 13 --spectra_start 200 --spectra_end 1200

"""
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Raman area-range viewer.")
    parser.add_argument("path", type=str, help="Path to the data directory.")
    parser.add_argument("x", type=int, help="Number of grid points in x.")
    parser.add_argument("y", type=int, help="Number of grid points in y.")
    parser.add_argument("--end_members", type=int, default=-1,
                        help="Number of end members to find (default -1, auto).")
    parser.add_argument("--spectra_start", type=float, default=None,
                        help="Spectra start position in cm⁻¹ (default: None).")
    parser.add_argument("--spectra_end", type=float, default=None,
                        help="Spectra end position in cm⁻¹ (default: None).")
    return parser.parse_args()

def load_data(path, x, y):
    # initialise the raman data object to use its methods
    raman_data = Raman_Data(Path(path).expanduser(), x, y)

    hsi_cube = raman_data.get_raw_hsi_cube()

    return hsi_cube

def unmix(hsi_cube):

    analysis = Analysis(hsi_cube)
    ns = analysis.ns

    print(f"Predicted vlaues of end members: 85% Variance in PCA: {ns[0]}, Elbow in PCA: {ns[1]}, Virtual Dimensionality (False Alarm Rate: 1e-5): {ns[2]}")
    n_endmembers, confidence = analysis.predicted_n, analysis.confidence

    nfindr = rp.analysis.unmix.NFINDR(n_endmembers=n_endmembers, abundance_method='fcls')
    abundance_maps, endmembers = nfindr.apply(hsi_cube)

    rp.plot.spectra(
    endmembers, hsi_cube.spectral_axis, 
    plot_type="single stacked", 
    label=[f"Endmember {i + 1}" for i in range(len(endmembers))]
    )
    plt.show()

    rp.plot.image(abundance_maps,
                title=[f"Component {i + 1}" for i in range(len(abundance_maps))]
                )

    plt.show()
"""
def build_figure(area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map):
    # ax_image is axis for the image, ax_spectra for the raman shift spectra
    fig, (ax_image, ax_spectra) = plt.subplots(2, 1, figsize=(8, 20), squeeze=True, gridspec_kw={"height_ratios": [5, 2]})
    # reserve space at the bottom for two slider rows
    fig.subplots_adjust(bottom=0.10)

    # raman image
    ax_image.set_title("Raman Image")
    ax_image.set_axis_off()

    # colour bar max and min
    v_min, v_max = np.min(area_by_region), np.max(area_by_region)

    # initialise imshow with RGBA so that set_data(rgba) works correctly later
    image = ax_image.imshow(apply_intensity_mask(area_by_region[:, :, 0], v_min, v_max), aspect="auto", origin="upper")

    # keep a detached ScalarMappable to drive the colorbar independently of the RGBA image
    scalar_mappable = cm.ScalarMappable(norm=Normalize(vmin=v_min, vmax=v_max), cmap="viridis")
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax_image)
    cbar.set_ticks(np.linspace(v_min, v_max, 5))

    # single spectra plot
    ax_spectra.set_title("Intensity Spectra")
    ax_spectra.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectra.set_ylabel("Intensity")

    # pixel_spectra is the spectra for that pixel, use pixel_map[0, 0] to match the initial heatmap view
    initial_pixel = pixel_map[0, 0]
    (pixel_specra,) = ax_spectra.plot(raman_shift, spectra_by_pixel[initial_pixel])

    # convert to numpy array
    raman_shift_arr = np.array(raman_shift)
    # lines indicating region of rolling window in the pixel's sepctra
    lower_limit_line = ax_spectra.axvline(raman_shift_arr[0], color="red", linestyle="--", alpha=0.7)
    upper_limit_line = ax_spectra.axvline(raman_shift_arr[idx_step], color="red", linestyle="--", alpha=0.7)

    # hover label
    hover_text = ax_image.text(
        0.01, 0.99, "", transform=ax_image.transAxes,
        va="top", ha="left", color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    # slider and textbox axes
    ax_slider = fig.add_axes([0.15, 0.055, 0.55, 0.020])
    ax_box = fig.add_axes([0.78, 0.055, 0.10, 0.020])
    ax_intensity_range = fig.add_axes([0.15, 0.020, 0.72, 0.020])

    # indices that match the number of area by regions
    indices = np.arange(area_by_region.shape[-1])

    # spectra window slider, this controls the area under rolling window being viewed
    spectra_window = Slider(
        ax=ax_slider, label="Raman Shift",
        valmin=0, valmax=indices[-1],
        valinit=0, valstep=indices,
    )
    spectra_window.valtext.set_text(str(raman_shift[0]))

    # initial text box to indicate pixel being viewed in the single spectra plot
    text_box = TextBox(ax_box, "Pixel:", textalignment="center")
    text_box.set_val(str(initial_pixel))

    # intensity range slider
    intensity_range = RangeSlider(
        ax=ax_intensity_range, label="Intensity",
        valmin=v_min, valmax=v_max,
        valinit=(v_min, v_max),
    )

    return (
        fig, ax_image, ax_spectra,
        image, pixel_specra,
        lower_limit_line, upper_limit_line,
        hover_text, spectra_window, text_box,
        raman_shift_arr,
        intensity_range, cbar, scalar_mappable,
        v_min, v_max,
    )

def make_update(fig, ax_spectra, image, pixel_specra,
                lower_limit_line, upper_limit_line,
                spectra_window, text_box, intensity_range,
                area_by_region, spectra_by_pixel,
                raman_shift_arr, idx_step, cbar, scalar_mappable):

    def update(_val):
        index = int(spectra_window.val)
        try:
            pixel = int(text_box.text)
        except (ValueError, KeyError):
            return

        # current intensity clip range
        i_min, i_max = intensity_range.val

        # mask data for current slice
        image.set_data(apply_intensity_mask(area_by_region[:, :, index], i_min, i_max))

        # update cbar ticks
        scalar_mappable.set_clim(i_min, i_max)
        cbar.set_ticks(np.linspace(i_min, i_max, 5))
        cbar.update_normal(scalar_mappable)

        new_y = spectra_by_pixel[pixel]
        pixel_specra.set_ydata(new_y)

        x_start = raman_shift_arr[index]
        x_end   = raman_shift_arr[index + idx_step - 1]
        lower_limit_line.set_xdata([x_start, x_start])
        upper_limit_line.set_xdata([x_end,   x_end])
        spectra_window.valtext.set_text(f"{x_start:.0f}-{x_end:.0f}")

        # set y limits with 10 percent padding on both sides
        ax_spectra.set_ylim(np.min(new_y) * 0.9, np.max(new_y) * 1.1)
        fig.canvas.draw_idle()

    return update

def make_on_hover(fig, ax_image, hover_text, pixel_map, raman_data):

    def on_hover(event):
        if event.inaxes == ax_image:
            col = np.clip(int(event.xdata + 0.5), 0, raman_data.y - 1)
            row = np.clip(int(event.ydata + 0.5), 0, raman_data.x - 1)
            hover_text.set_text(f"Pixel {pixel_map[row, col]}")
        else:
            hover_text.set_text("")
        fig.canvas.draw_idle()

    return on_hover

def make_on_click(fig, ax_image, ax_spectra,
                  pixel_specra, text_box,
                  spectra_window, spectra_by_pixel,
                  pixel_map, raman_data):

    def on_click(event):
        if event.inaxes != ax_image or not event.dblclick:
            return

        col   = np.clip(int(event.xdata + 0.5), 0, raman_data.y - 1)
        row   = np.clip(int(event.ydata + 0.5), 0, raman_data.x - 1)
        pixel = pixel_map[row, col]

        text_box.set_val(str(pixel)) # also fires on_submit → update()

        new_y = spectra_by_pixel[pixel]
        pixel_specra.set_ydata(new_y)
        ax_spectra.set_ylim(np.min(new_y) * 0.9, np.max(new_y) * 1.1)
        fig.canvas.draw_idle()

    return on_click
"""
def main(path, x, y, spectra_start, spectra_end):
    

    hsi_cube = load_data(path, x, y)

    preprocessing_pipeline = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=(spectra_start, spectra_end))
    ])
    
    # cropping 
    hsi_cube = preprocessing_pipeline.apply(hsi_cube)

    unmix(hsi_cube)


if __name__ == "__main__":
    args = parse_args()
    main(args.path, args.x, args.y, args.spectra_start, args.spectra_end)