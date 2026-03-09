from pathlib import Path
from raman_helper import Raman_Data
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, TextBox, RangeSlider
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

"""
e.g. input:
python3 plot_raman.py ~/Code/Data_SH/FullCavity_20x20_2umsteps 20 20 --pipeline 2 --rolling_window 50

python3 plot_raman.py ~/Code/Data_SH/SB008 10 13 --pipeline 0 --rolling_window 50

python3 plot_raman.py ~/Code/Data_SH/SB008 10 13 --pipeline 0 --rolling_window 50 --spectra_start 0 --spectra_end 100 

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Raman area-range viewer.")
    parser.add_argument("path", type=str, help="Path to the data directory.")
    parser.add_argument("x", type=int, help="Number of grid points in x.")
    parser.add_argument("y", type=int, help="Number of grid points in y.")
    parser.add_argument("--pipeline", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Preprocessing pipeline (default: 1).")
    parser.add_argument("--rolling_window", type=float, default=20,
                        help="Rolling window width in cm⁻¹ (default: 20).")
    parser.add_argument("--spectra_start", type=float, default=None,
                        help="Spectra Range window width in cm⁻¹ (default: None).")
    parser.add_argument("--spectra_end", type=float, default=None,
                        help="Spectra Range window width in cm⁻¹ (default: None).")
    return parser.parse_args()

def apply_intensity_mask(data_2d, i_min, i_max):
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=i_min, vmax=i_max, clip=True)
    rgba = cmap(norm(data_2d))
    outside = (data_2d < i_min) | (data_2d > i_max) # mask pixels out the intensity range to grey
    rgba[outside] = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba

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

        pixel_specra.set_ydata(spectra_by_pixel[pixel])

        x_start = raman_shift_arr[index]
        x_end = raman_shift_arr[index + idx_step - 1]
        lower_limit_line.set_xdata([x_start, x_start])
        upper_limit_line.set_xdata([x_end, x_end])
        spectra_window.valtext.set_text(f"{x_start:.0f}-{x_end:.0f}")

        ax_spectra.relim()
        ax_spectra.autoscale_view()
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
                  pixel_specra, text_box, spectra_by_pixel,
                  pixel_map, raman_data, intensity_range):

    def on_click(event):
        if event.inaxes != ax_image or not event.dblclick:
            return

        col   = np.clip(int(event.xdata + 0.5), 0, raman_data.y - 1)
        row   = np.clip(int(event.ydata + 0.5), 0, raman_data.x - 1)
        pixel = pixel_map[row, col]

        text_box.set_val(str(pixel)) # also fires on_submit → update()

        pixel_specra.set_ydata(spectra_by_pixel[pixel])
        ax_spectra.relim()
        ax_spectra.autoscale_view()
        fig.canvas.draw_idle()

    return on_click

def main(path, x, y, pipeline, rolling_window, spectra_start, spectra_end):
    
    raman_data = Raman_Data(Path(path).expanduser(), x, y)
    
    area_cube, spectra_of_pixel, raman_shift, idx_step, pixel_map = raman_data.get_hsi_cube(pipeline, rolling_window, spectra_start, spectra_end)

    fig, (ax_image, ax_spectra) = plt.subplots(2, 1, figsize=(8, 20), squeeze=True, gridspec_kw={"height_ratios": [5, 2]})
    fig.subplots_adjust(bottom=0.20)

    ax_image.set_title("Raman Image")
    ax_image.set_axis_off()
    ax_spectra.set_title("Intensity Spectra")
    ax_spectra.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectra.set_ylabel("Intensity")
    
    ax_slider = fig.add_axes([0.15, 0.055, 0.55, 0.020])
    ax_box = fig.add_axes([0.78, 0.055, 0.10, 0.020])
    ax_intensity_range = fig.add_axes([0.15, 0.020, 0.72, 0.020])
    
    # colour bar max and min
    v_min, v_max = np.min(area_cube), np.max(area_cube)

    # image is heatmap of current region, pixel_spectra is spectra of initially first pixel
    image = ax_image.imshow(apply_intensity_mask(area_cube[:, :, 0], v_min, v_max), aspect="auto", origin="upper")
    (pixel_specra,) = ax_spectra.plot(raman_shift, spectra_of_pixel[pixel_map[0, 0]])

    # keep a detached ScalarMappable to drive the colorbar independently of the RGBA image
    scalar_mappable = cm.ScalarMappable(norm=Normalize(vmin=v_min, vmax=v_max), cmap="viridis")
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax_image)
    cbar.set_ticks(np.linspace(v_min, v_max, 5))

    raman_shift_arr = np.array(raman_shift)
    indices = np.arange(area_cube.shape[-1])

    # spectra slider, intensity slider, textbox for pixel selection
    spectra_window = Slider(ax=ax_slider, label="Raman Shift", 
                            valmin=0, valmax=indices[-1],
                            valinit=0, valstep=indices
                            )
    spectra_window.valtext.set_text(str(raman_shift[0]))

    intensity_range = RangeSlider(ax=ax_intensity_range, label="Intensity",
                                  valmin=v_min, valmax=v_max,
                                  valinit=(v_min, v_max)
                                  )
    
    text_box = TextBox(ax_box, "Pixel:", textalignment="center")
    text_box.set_val(str(1))

    # lines indicating region of rolling window in the pixel's sepctra
    lower_limit_line = ax_spectra.axvline(raman_shift_arr[0], color="red", linestyle="--", alpha=0.7)
    upper_limit_line = ax_spectra.axvline(raman_shift_arr[idx_step], color="red", linestyle="--", alpha=0.7)

    # hover label
    hover_text = ax_image.text(
        0.01, 0.99, "", transform=ax_image.transAxes,
        va="top", ha="left", color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    update = make_update(
        fig, ax_spectra, image, pixel_specra,
        lower_limit_line, upper_limit_line,
        spectra_window, text_box, intensity_range,
        area_cube, spectra_of_pixel,
        raman_shift_arr, idx_step, cbar, scalar_mappable,
    )
    on_hover = make_on_hover(fig, ax_image, hover_text, pixel_map, raman_data)
    on_click = make_on_click(
        fig, ax_image, ax_spectra,
        pixel_specra, text_box, spectra_of_pixel,
        pixel_map, raman_data, intensity_range
    )

    text_box.on_submit(update)
    spectra_window.on_changed(update)
    intensity_range.on_changed(update)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args.path, args.x, args.y, args.pipeline, 
         args.rolling_window, args.spectra_start, args.spectra_end)