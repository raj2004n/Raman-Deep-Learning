import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .theme import *
from matplotlib.widgets import Slider, TextBox, RangeSlider, Button

def _apply_intensity_mask(image, i_min, i_max):
    """Applies intensity mask and greys out values outside the range.

    Args:
        image (_type_): Image 
        i_min (_type_): Minimum intensity value
        i_max (_type_): Maximum intensity value

    Returns:
        _type_: Masked image.
    """
    
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=i_min, vmax=i_max, clip=True)
    rgba = cmap(norm(image))

    outside = (image < i_min) | (image > i_max)
    rgba[outside] = [0.15, 0.15, 0.18, 1]
    return rgba

def show_hsi_viewer(auc_cube, spectra_list, raman_shift, idx_step, pixel_map, x, y):
    apply_theme()
    # mutable state for scale mode
    state = {"ln_scale": False}

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    
    # left
    ax_image        = fig.add_axes([0.09, 0.56, 0.65, 0.40])
    ax_spectrum     = fig.add_axes([0.09, 0.33, 0.65, 0.18])
    ax_ln_spectrum  = fig.add_axes([0.09, 0.07, 0.65, 0.18])

    # bottom
    ax_slider = fig.add_axes([0.09, 0.01, 0.65, 0.035])

    # right
    ax_intensity_slider = fig.add_axes([0.8, 0.32, 0.03, 0.63])
    ax_button           = fig.add_axes([0.8, 0.22, 0.16, 0.06])
    ax_box              = fig.add_axes([0.8, 0.14, 0.16, 0.05])

    ax_image.set_title("Raman Image")
    ax_image.set_axis_off()
    ax_spectrum.set_title("Intensity Spectra")
    ax_spectrum.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectrum.set_ylabel("Intensity")
    ax_ln_spectrum.set_title("Intensity Spectra (ln scale)")
    ax_ln_spectrum.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_ln_spectrum.set_ylabel("ln(Intensity)")

    i_min, i_max = np.min(auc_cube), np.max(auc_cube)
    for ax in fig.axes:
        ax.set_facecolor((1,0,0,0.05))

    correction = abs(min(i_min, 0))
    if correction > 0:
        print(f"Intensity correction applied: +{correction:.4f} to shift all values positive")

    auc_cube_corrected  = auc_cube + correction + 1e-10
    log_v_min           = np.log(np.min(auc_cube_corrected))
    log_v_max           = np.log(np.max(auc_cube_corrected))

    # initial heatmap and spectra
    image                   = ax_image.imshow(_apply_intensity_mask(auc_cube[:, :, 0], i_min, i_max), aspect="equal", origin="upper")
    first_spectrum          = spectra_list[pixel_map[0, 0]]
    (pixel_spectrum,)       = ax_spectrum.plot(raman_shift, first_spectrum, color=ACCENT)
    (pixel_ln_spectrum,)    = ax_ln_spectrum.plot(raman_shift, np.log(np.maximum(first_spectrum + correction + 1e-10, 1e-10)), color=ACCENT)

    scalar_mappable = cm.ScalarMappable(norm=Normalize(i_min, i_max), cmap="magma")
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax_image)
    cbar.set_ticks(np.linspace(i_min, i_max, 5))
    cbar.ax.yaxis.set_tick_params(color=FG, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)

    raman_shift_arr = np.array(raman_shift)
    indices = np.arange(auc_cube.shape[-1])

    rolling_window = Slider(
        ax=ax_slider, label="Raman Shift",
        valmin=0, valmax=indices[-1],
        valinit=0, valstep=indices
    )
    rolling_window.label.set_color(FG)
    rolling_window.valtext.set_color(FG)
    rolling_window.track.set_color(SLIDER_TRACK)
    rolling_window.poly.set_color(SLIDER_ACTIVE)
    rolling_window.valtext.set_text(f"{raman_shift[0]:.0f} cm⁻¹")
    
    ax_slider.set_facecolor(WIDGET_PANEL)

    # start slider in normal scale
    intensity_slider = RangeSlider(
        ax=ax_intensity_slider, label="Intensity",
        valmin=i_min, valmax=i_max,
        valinit=(i_min, i_max),
        orientation="vertical",
    )
    intensity_slider.label.set_color(FG)
    intensity_slider.valtext.set_color(FG)
    intensity_slider.track.set_color(SLIDER_TRACK)
    intensity_slider.poly.set_color(SLIDER_ACTIVE)
    ax_intensity_slider.set_facecolor(WIDGET_PANEL)

    scale_button = Button(ax_button, "Switch to ln scale", color=BUTTON_COLOR, hovercolor=BUTTON_HOVER)
    scale_button.label.set_color(FG)
    scale_button.label.set_fontsize(8)
    ax_button.set_facecolor(WIDGET_PANEL)

    text_box = TextBox(
        ax_box,
        "Pixel:",
        textalignment="center",
        color=WIDGET_SURFACE,
        hovercolor=WIDGET_EDGE
    )
    text_box.label.set_color(FG)
    text_box.text_disp.set_color(FG)
    text_box.set_val(str(pixel_map[0, 0]))
    ax_box.set_facecolor(WIDGET_PANEL)

    lower_limit_line    = ax_spectrum.axvline(raman_shift_arr[0], color="#E07B54", linestyle="--", alpha=0.7)
    upper_limit_line    = ax_spectrum.axvline(raman_shift_arr[idx_step], color="#E07B54", linestyle="--", alpha=0.7)
    lower_ln_limit_line = ax_ln_spectrum.axvline(raman_shift_arr[0], color="#E07B54", linestyle="--", alpha=0.7)
    upper_ln_limit_line = ax_ln_spectrum.axvline(raman_shift_arr[idx_step], color="#E07B54", linestyle="--", alpha=0.7)

    hover_text = ax_image.text(
        0.01, 0.99, "", transform=ax_image.transAxes,
        va="top", ha="left", color=FG, fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=GRID, edgecolor=FG, alpha=0.85),
    )
    
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    def _get_image_range():
        """Returns (i_min, i_max) in original scale regardless of slider mode."""
        v_min, v_max = intensity_slider.val
        if state["ln_scale"]:
            return np.exp(v_min) - correction, np.exp(v_max) - correction
        return v_min, v_max

    def update(_val):
        index = int(rolling_window.val)
        try:
            pixel = int(text_box.text)
        except (ValueError, KeyError):
            return

        i_min_cur, i_max_cur = _get_image_range()

        image.set_data(_apply_intensity_mask(auc_cube[:, :, index], i_min_cur, i_max_cur))
        scalar_mappable.set_clim(i_min_cur, i_max_cur)
        cbar.set_ticks(np.linspace(i_min_cur, i_max_cur, 5))
        cbar.update_normal(scalar_mappable)

        spectrum = spectra_list[pixel]
        pixel_spectrum.set_ydata(spectrum)
        pixel_ln_spectrum.set_ydata(np.log(np.maximum(spectrum + correction + 1e-10, 1e-10)))

        x_start = raman_shift_arr[index]
        x_end   = raman_shift_arr[index + idx_step - 1]
        for line in [lower_limit_line, lower_ln_limit_line]:
            line.set_xdata([x_start, x_start])
        for line in [upper_limit_line, upper_ln_limit_line]:
            line.set_xdata([x_end, x_end])
        rolling_window.valtext.set_text(f"{x_start:.0f}-{x_end:.0f} cm⁻¹")
        
        ax_spectrum.relim()
        ax_spectrum.autoscale_view()
        ax_ln_spectrum.relim()
        ax_ln_spectrum.autoscale_view()

        # recapture background after limit changes, then blit
        fig.canvas.restore_region(background)
        for ax in [ax_image, ax_spectrum, ax_ln_spectrum]:
            ax.redraw_in_frame()
        fig.canvas.blit(fig.bbox)

    def on_resize(_event):
        nonlocal background
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)

    fig.canvas.mpl_connect("resize_event", on_resize)

    def on_scale_toggle(_event):
        v_min_cur, v_max_cur = intensity_slider.val
        state["ln_scale"] = not state["ln_scale"]

        if state["ln_scale"]:
            new_min = np.log(max(v_min_cur + correction + 1e-10, 1e-10))
            new_max = np.log(max(v_max_cur + correction + 1e-10, 1e-10))
            intensity_slider.ax.set_ylim(log_v_min, log_v_max)
            intensity_slider.valmin = log_v_min
            intensity_slider.valmax = log_v_max
            intensity_slider.label.set_text("ln(Intensity)")
            scale_button.label.set_text("Switch to standard scale")
        else:
            new_min = np.exp(v_min_cur) - correction
            new_max = np.exp(v_max_cur) - correction
            intensity_slider.ax.set_ylim(i_min, i_max)
            intensity_slider.valmin = i_min
            intensity_slider.valmax = i_max
            intensity_slider.label.set_text("Intensity")
            scale_button.label.set_text("Switch to ln scale")

        intensity_slider.set_val((new_min, new_max))
        fig.canvas.draw_idle()

    def on_hover(event):
        new_text = ""
        if event.inaxes == ax_image:
            col = np.clip(int(event.xdata + 0.5), 0, y - 1)
            row = np.clip(int(event.ydata + 0.5), 0, x - 1)
            new_text = f"Pixel {pixel_map[row, col]}"
        if hover_text.get_text() != new_text:
            hover_text.set_text(new_text)
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_image or not event.dblclick:
            return
        col     = np.clip(int(event.xdata + 0.5), 0, y - 1)
        row     = np.clip(int(event.ydata + 0.5), 0, x - 1)
        pixel   = pixel_map[row, col]
        text_box.set_val(str(pixel))

        spectrum = spectra_list[pixel]
        pixel_spectrum.set_ydata(spectrum)
        pixel_ln_spectrum.set_ydata(np.log(np.maximum(spectrum + correction + 1e-10, 1e-10)))

        ax_spectrum.relim()
        ax_spectrum.autoscale_view()
        ax_ln_spectrum.relim()
        ax_ln_spectrum.autoscale_view()
        fig.canvas.draw_idle()

    text_box.on_submit(update)
    rolling_window.on_changed(update)
    intensity_slider.on_changed(update)
    scale_button.on_clicked(on_scale_toggle)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()