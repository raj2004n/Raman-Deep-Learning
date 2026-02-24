import argparse
from pathlib import Path

import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from raman_helper import Raman_Data

"""
e.g. input:
python3 plot_raman.py ~/Code/Data_SH/FullCavity_20x20_2umsteps 20 20 --pipeline 2 --rolling-window 50

python3 plot_raman.py ~/Code/Data_SH/SB008 10 13 --pipeline 0 --rolling-window 50

python3 plot_raman.py ~/Code/Data_SH/SB008 10 13 --pipeline 0 --rolling-window 50 --spectra-start 0 --spectra-end 100 

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Raman area-range viewer."
    )
    parser.add_argument("path", type=str, help="Path to the data directory.")

    parser.add_argument("x", type=int, help="Number of grid points in x.")

    parser.add_argument("y", type=int, help="Number of grid points in y.")

    parser.add_argument("--pipeline", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Preprocessing pipeline (default: 1).")
    
    parser.add_argument("--rolling-window", type=float, default=20,
                        help="Rolling window width in cm⁻¹ (default: 20).")
    
    parser.add_argument("--spectra-start", type=float, default=None,
                        help="Spectra Range window width in cm⁻¹ (default: None).")
    
    parser.add_argument("--spectra-end", type=float, default=None,
                        help="Spectra Range window width in cm⁻¹ (default: None).")
    
    return parser.parse_args()

def load_data(path, x, y, pipeline, rolling_window, spectra_start, spectra_end):
    # initialise the raman data object to use its methods
    raman_data = Raman_Data(Path(path).expanduser(), x, y)

    # get data to visualise
    area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map = raman_data.get_area_range(pipeline, rolling_window, spectra_start, spectra_end)

    return raman_data, area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map

def build_figure(area_by_region, spectra_by_pixel, raman_shift, idx_step):
    # ax_image is axis for the image, ax_spectra for the raman shift spectra
    fig, (ax_image, ax_spectra) = plt.subplots(2, 1, figsize=(8, 20), squeeze=True, gridspec_kw={"height_ratios": [5, 2]})

    # raman image
    ax_image.set_title("Raman Image")
    ax_image.set_axis_off()
    rp.plot.image(area_by_region[:, :, 0], ax=ax_image)
    image = ax_image.get_images()[0]

    # colour bar max and min
    v_min, v_max = np.min(area_by_region), np.max(area_by_region)
    image.set_clim(v_min, v_max)
    cbar = ax_image.images[0].colorbar
    # just want to see the first and last val of colour bar
    cbar.set_ticks(np.linspace(v_min, v_max, 2))

    # single spectra plot
    ax_spectra.set_title("Intensity Spectra")
    ax_spectra.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectra.set_ylabel("Intensity")
    
    # pixel_spectra is the spectra for that pixel
    (pixel_specra,) = ax_spectra.plot(raman_shift, spectra_by_pixel[1])

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
    ax_slider = fig.add_axes([0.15, 0.02, 0.5, 0.04])
    ax_box = fig.add_axes([0.9, 0.02, 0.07, 0.04])

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
    text_box.set_val("1")

    return (
        fig, ax_image, ax_spectra,
        image, pixel_specra,
        lower_limit_line, upper_limit_line,
        hover_text, spectra_window, text_box,
        raman_shift_arr,
    )

def make_update(fig, ax_spectra, image, pixel_specra,
                lower_limit_line, upper_limit_line,
                spectra_window, text_box,
                area_by_region, spectra_by_pixel,
                raman_shift_arr, idx_step):

    def update(_val):
        index = int(spectra_window.val)
        try:
            pixel = int(text_box.text)
        except (ValueError, KeyError):
            return

        image.set_data(area_by_region[:, :, index])

        new_y = spectra_by_pixel[pixel]
        pixel_specra.set_ydata(new_y)

        x_start = raman_shift_arr[index]
        x_end   = raman_shift_arr[index + idx_step - 1]
        lower_limit_line.set_xdata([x_start, x_start])
        upper_limit_line.set_xdata([x_end,   x_end])
        spectra_window.valtext.set_text(f"{x_start:.0f}-{x_end:.0f}")

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

def main():
    args = parse_args()

    raman_data, area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map = (
        load_data(args.path, args.x, args.y, args.pipeline, args.rolling_window, args.spectra_start, args.spectra_end)
    )

    (
        fig, ax_image, ax_spectra,
        image, pixel_specra,
        lower_limit_line, upper_limit_line,
        hover_text, spectra_window, text_box,
        raman_shift_arr,
    ) = build_figure(area_by_region, spectra_by_pixel, raman_shift, idx_step)

    update = make_update(
        fig, ax_spectra, image, pixel_specra,
        lower_limit_line, upper_limit_line,
        spectra_window, text_box,
        area_by_region, spectra_by_pixel,
        raman_shift_arr, idx_step,
    )
    on_hover = make_on_hover(fig, ax_image, hover_text, pixel_map, raman_data)
    on_click = make_on_click(
        fig, ax_image, ax_spectra,
        pixel_specra, text_box,
        spectra_window, spectra_by_pixel,
        pixel_map, raman_data,
    )

    text_box.on_submit(update)
    spectra_window.on_changed(update)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()


if __name__ == "__main__":
    main()