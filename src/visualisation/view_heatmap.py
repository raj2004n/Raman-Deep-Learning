import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, TextBox, RangeSlider

def _apply_intensity_mask(image, i_min, i_max):
    """Applies intensity mask and greys out values outside the range.

    Args:
        image (_type_): Image 
        i_min (_type_): Minimum intensity value
        i_max (_type_): Maximum intensity value

    Returns:
        _type_: Masked image.
    """
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=i_min, vmax=i_max, clip=True)
    rgba = cmap(norm(image))

    # mask pixels out the intensity range to grey
    outside = (image < i_min) | (image > i_max)
    rgba[outside] = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba

#TODO: better name than area cube come on
#TODO: option to use ln scaler, or no scaler
def show_hsi_viewer(auc_cube, spectra_list, raman_shift, idx_step, pixel_map, x, y):
    fig, (ax_image, ax_spectrum, ax_spectrum_log) = plt.subplots(3, 1, figsize=(10, 13), squeeze=True, gridspec_kw={"height_ratios": [4, 2, 2]})
    fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.18, hspace=0.45)

    # axes for widgets
    ax_slider = fig.add_axes([0.12, 0.10, 0.55, 0.02])
    ax_box = fig.add_axes([0.78, 0.10, 0.10, 0.02])
    ax_intensity_range = fig.add_axes([0.12, 0.05, 0.72, 0.02])
    ax_image.set_title("Raman Image")

    ax_image.set_axis_off()
    ax_spectrum.set_title("Intensity Spectra")
    ax_spectrum.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectrum.set_ylabel("Intensity")
    ax_spectrum_log.set_title("Intensity Spectra (ln scale)")
    ax_spectrum_log.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectrum_log.set_ylabel("ln(Intensity)")

    # colourbar range from full data — log scaled
    i_min, i_max = np.min(auc_cube), np.max(auc_cube)

    correction = abs(min(i_min, 0))
    if correction > 0:
        print(f"Intensity correction applied: +{correction:.4f} to shift all values positive")
    
    auc_cube_corrected = auc_cube + correction + 1e-10 # to aviod ln(0)
    i_min_corrected = np.min(auc_cube_corrected)
    i_max_corrected = np.max(auc_cube_corrected)

    log_v_min = np.log(i_min_corrected)
    log_v_max = np.log(i_max_corrected)

    # initial heatmap and spectrums
    image                   = ax_image.imshow(_apply_intensity_mask(auc_cube[:, :, 0], i_min, i_max), aspect="equal", origin="upper")
    first_spectrum          = spectra_list[pixel_map[0, 0]]
    (pixel_spectrum,)       = ax_spectrum.plot(raman_shift, first_spectrum)
    (pixel_spectrum_log,)   = ax_spectrum_log.plot(raman_shift, np.log(first_spectrum + correction))

    # detached ScalarMappable drives the colorbar independently of the RGBA image
    scalar_mappable = cm.ScalarMappable(norm=Normalize(i_min, i_max), cmap="viridis")    
    scalar_mappable.set_array([])

    cbar = fig.colorbar(scalar_mappable, ax=ax_image)
    cbar.set_ticks(np.linspace(i_min, i_max, 5))

    raman_shift_arr = np.array(raman_shift)
    indices = np.arange(auc_cube.shape[-1])

    # widgets
    rolling_window = Slider(
        ax=ax_slider, label="Raman Shift",
        valmin=0, valmax=indices[-1],
        valinit=0, valstep=indices
    )
    rolling_window.valtext.set_text(str(raman_shift[0]))

    intensity_range = RangeSlider(
        ax=ax_intensity_range, label="ln(Intensity)",
        valmin=log_v_min, valmax=log_v_max,
        valinit=(log_v_min, log_v_max)
    )

    text_box = TextBox(ax_box, "Pixel:", textalignment="center")
    text_box.set_val(str(130))

    # lines on spectrum indicating region rolling window being viewed on image
    lower_limit_line        = ax_spectrum.axvline(raman_shift_arr[0], color="red", linestyle="--", alpha=0.7)
    upper_limit_line        = ax_spectrum.axvline(raman_shift_arr[idx_step], color="red", linestyle="--", alpha=0.7)
    lower_limit_line_log    = ax_spectrum_log.axvline(raman_shift_arr[0], color="red", linestyle="--", alpha=0.7)
    upper_limit_line_log    = ax_spectrum_log.axvline(raman_shift_arr[idx_step], color="red", linestyle="--", alpha=0.7)

    # pixel label shown on hover
    hover_text = ax_image.text(
        0.01, 0.99, "", transform=ax_image.transAxes,
        va="top", ha="left", color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    def update(_val):
        index = int(rolling_window.val)
        try:
            pixel = int(text_box.text)
        except (ValueError, KeyError):
            return

        # get intensity range and convert to normal scale
        log_i_min, log_i_max = intensity_range.val
        i_min = np.exp(log_i_min) - correction  # convert back to original scale for mask
        i_max = np.exp(log_i_max) - correction
        
        # update image, cbar, pixel spectrums
        image.set_data(_apply_intensity_mask(auc_cube[:, :, index], i_min, i_max))

        scalar_mappable.set_clim(i_min, i_max)
        cbar.set_ticks(np.linspace(i_min, i_max, 5))
        cbar.update_normal(scalar_mappable)

        spectrum = spectra_list[pixel]
        pixel_spectrum.set_ydata(spectrum)
        pixel_spectrum_log.set_ydata(np.log(spectrum + correction + 1e-10))

        x_start = raman_shift_arr[index]
        x_end   = raman_shift_arr[index + idx_step - 1]

        for line in [lower_limit_line, lower_limit_line_log]:
            line.set_xdata([x_start, x_start])
        for line in [upper_limit_line, upper_limit_line_log]:
            line.set_xdata([x_end, x_end])
        rolling_window.valtext.set_text(f"{x_start:.0f}-{x_end:.0f}")

        ax_spectrum.relim()
        ax_spectrum.autoscale_view()
        ax_spectrum_log.relim()
        ax_spectrum_log.autoscale_view()
        fig.canvas.draw_idle()

    def on_hover(event):
        if event.inaxes == ax_image:
            col = np.clip(int(event.xdata + 0.5), 0, y - 1)
            row = np.clip(int(event.ydata + 0.5), 0, x - 1)
            hover_text.set_text(f"Pixel {pixel_map[row, col]}")
        else:
            hover_text.set_text("")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_image or not event.dblclick:
            return
        col   = np.clip(int(event.xdata + 0.5), 0, y - 1)
        row   = np.clip(int(event.ydata + 0.5), 0, x - 1)
        pixel = pixel_map[row, col]
        text_box.set_val(str(pixel))

        spectrum = spectra_list[pixel]
        pixel_spectrum.set_ydata(spectrum)
        pixel_spectrum_log.set_ydata(np.log(spectrum + correction + 1e-10))

        ax_spectrum.relim()
        ax_spectrum.autoscale_view()
        ax_spectrum_log.relim()
        ax_spectrum_log.autoscale_view()
        fig.canvas.draw_idle()

    # wire up callbacks
    text_box.on_submit(update)
    rolling_window.on_changed(update)
    intensity_range.on_changed(update)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()