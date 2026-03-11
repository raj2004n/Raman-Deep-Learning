import numpy as np
import ramanspy as rp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, TextBox, RangeSlider
from src.analysis.endmember_estimator import estimate_endmembers

def apply_intensity_mask(data_2d, i_min, i_max):
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=i_min, vmax=i_max, clip=True)
    rgba = cmap(norm(data_2d))
    outside = (data_2d < i_min) | (data_2d > i_max) # mask pixels out the intensity range to grey
    rgba[outside] = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba

def _make_update(fig, ax_spectrum, image, pixel_spectrum,
                lower_limit_line, upper_limit_line,
                rolling_window, text_box, intensity_range,
                area_by_region, spectra_by_pixel,
                raman_shift_arr, idx_step, cbar, scalar_mappable):

    def update(_val):
        index = int(rolling_window.val)
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

        pixel_spectrum.set_ydata(spectra_by_pixel[pixel])

        x_start = raman_shift_arr[index]
        x_end = raman_shift_arr[index + idx_step - 1]
        lower_limit_line.set_xdata([x_start, x_start])
        upper_limit_line.set_xdata([x_end, x_end])
        rolling_window.valtext.set_text(f"{x_start:.0f}-{x_end:.0f}")

        ax_spectrum.relim()
        ax_spectrum.autoscale_view()
        fig.canvas.draw_idle()

    return update

def _make_on_hover(fig, ax_image, hover_text, pixel_map, x, y):

    def on_hover(event):
        if event.inaxes == ax_image:
            col = np.clip(int(event.xdata + 0.5), 0, y - 1)
            row = np.clip(int(event.ydata + 0.5), 0, x - 1)
            hover_text.set_text(f"Pixel {pixel_map[row, col]}")
        else:
            hover_text.set_text("")
        fig.canvas.draw_idle()

    return on_hover

def _make_on_click(fig, ax_image, ax_spectra,
                  pixel_specra, text_box, spectra_by_pixel,
                  pixel_map, x, y):

    def on_click(event):
        if event.inaxes != ax_image or not event.dblclick:
            return

        col   = np.clip(int(event.xdata + 0.5), 0, y - 1)
        row   = np.clip(int(event.ydata + 0.5), 0, x - 1)
        pixel = pixel_map[row, col]

        text_box.set_val(str(pixel)) # also fires on_submit → update()

        pixel_specra.set_ydata(spectra_by_pixel[pixel])
        ax_spectra.relim()
        ax_spectra.autoscale_view()
        fig.canvas.draw_idle()

    return on_click

def show_hsi_viewer(area_cube, spectrum_of_pixel, raman_shift, idx_step, pixel_map, x, y):
    fig, (ax_image, ax_spectrum) = plt.subplots(
        2, 1, figsize=(8, 20), squeeze=True, 
        gridspec_kw={"height_ratios": [5, 2]}
        )
    fig.subplots_adjust(bottom=0.20)

    ax_image.set_title("Raman Image")
    ax_image.set_axis_off()
    ax_spectrum.set_title("Intensity Spectra")
    ax_spectrum.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectrum.set_ylabel("Intensity")

    # axes for widgets
    ax_slider = fig.add_axes([0.15, 0.055, 0.55, 0.020])
    ax_box = fig.add_axes([0.78, 0.055, 0.10, 0.020])
    ax_intensity_range = fig.add_axes([0.15, 0.020, 0.72, 0.020])

    # colourbar range from full data
    v_min, v_max = np.min(area_cube), np.max(area_cube)

    # initial heatmap and spectra line
    image = ax_image.imshow(
        apply_intensity_mask(area_cube[:, :, 0], v_min, v_max),
        aspect="auto", origin="upper"
    )
    # first spectrum of pixel one
    (pixel_spectrum,) = ax_spectrum.plot(raman_shift, spectrum_of_pixel[pixel_map[0, 0]])

    # detached ScalarMappable drives the colorbar independently of the RGBA image
    scalar_mappable = cm.ScalarMappable(norm=Normalize(vmin=v_min, vmax=v_max), cmap="viridis")
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax_image)
    cbar.set_ticks(np.linspace(v_min, v_max, 5))

    raman_shift_arr = np.array(raman_shift)
    indices = np.arange(area_cube.shape[-1])

    # widgets
    rolling_window = Slider(
        ax=ax_slider, label="Raman Shift",
        valmin=0, valmax=indices[-1],
        valinit=0, valstep=indices
    )
    rolling_window.valtext.set_text(str(raman_shift[0]))

    intensity_range = RangeSlider(
        ax=ax_intensity_range, label="Intensity",
        valmin=v_min, valmax=v_max,
        valinit=(v_min, v_max)
    )

    text_box = TextBox(ax_box, "Pixel:", textalignment="center")
    text_box.set_val(str(1))

    # lines on spectrum indicating region rolling window being viewed on image
    lower_limit_line = ax_spectrum.axvline(raman_shift_arr[0], color="red", linestyle="--", alpha=0.7)
    upper_limit_line = ax_spectrum.axvline(raman_shift_arr[idx_step], color="red", linestyle="--", alpha=0.7)

    # pixel label shown on hover
    hover_text = ax_image.text(
        0.01, 0.99, "", transform=ax_image.transAxes,
        va="top", ha="left", color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    # build callbacks
    update = _make_update(
        fig, ax_spectrum, image, pixel_spectrum,
        lower_limit_line, upper_limit_line,
        rolling_window, text_box, intensity_range,
        area_cube, spectrum_of_pixel,
        raman_shift_arr, idx_step, cbar, scalar_mappable,
    )
    on_hover = _make_on_hover(fig, ax_image, hover_text, pixel_map, x, y)
    on_click = _make_on_click(
        fig, ax_image, ax_spectrum,
        pixel_spectrum, text_box, spectrum_of_pixel,
        pixel_map, x, y
    )

    # wire up callbacks
    text_box.on_submit(update)
    rolling_window.on_changed(update)
    intensity_range.on_changed(update)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

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