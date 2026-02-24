from raman_helper import *
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

path = Path("~/Code/Data_SH/SB008").expanduser()
raman_data = Raman_Data(path, 10, 13)

preprocessed_slice, raw_slice, raman_shifts = raman_data.get_all_spectra(1)

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(10, 10, figure=fig, hspace=0.6)

ax_raw= fig.add_subplot(gs[0:4, :])

ax_spectra = fig.add_subplot(gs[5:9, :])

ax_box = fig.add_subplot(gs[9, 4:6])

# plot pixel of slice
rp.plot.spectra(raw_slice[1], title='Raw', ax=ax_raw)
rp.plot.spectra(preprocessed_slice[1], title='Preprocessed', ax=ax_spectra)
line_raw = ax_raw.get_lines()[0]
line_pre = ax_spectra.get_lines()[0]

def update(val):
    try:
        pixel = int(text_box.text)
    except ValueError:
        return # ignore
    
    # update spectra for pixel
    line_pre.set_data(raman_shifts, preprocessed_slice[pixel].spectral_data)
    line_raw.set_data(raman_shifts, raw_slice[pixel].spectral_data)

    ax_spectra.relim()
    ax_spectra.autoscale_view(scalex=False, scaley=True)
    ax_raw.relim()
    ax_raw.autoscale_view(scalex=False, scaley=True)
    fig.canvas.draw_idle()

# text box to enter pixel
text_box = TextBox(ax_box, "Pixel: ", initial='1', textalignment="center")
text_box.on_submit(update)
plt.show()
