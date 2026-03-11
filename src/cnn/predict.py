import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from data import standardise_data
from src.data.raman_helper import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load saved artefacts
model = keras.models.load_model("raman_cnn_model.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
wavenumber_range = np.load("wavenumber_range.npy")
x_min, x_max = wavenumber_range[0], wavenumber_range[1]

# Load hyperspectral image
path = Path("~/Code/Data_SH/SB008").expanduser()
raman_data = Raman_Data(path, 10, 13)
hsi_cube = raman_data.get_raw_hsi_cube()

length, width = hsi_cube.shape

# Predict label and top 5 probabilities for every pixel
predicted_labels_map = []
predicted_top5_map    = []

for l in range(length):
    spectra_list = list(hsi_cube[l])
    x_new        = standardise_data(spectra_list, target_length=913, x_min=x_min, x_max=x_max)
    predictions  = model.predict(x_new, verbose=0)  # (width, num_classes)

    # Top prediction
    predicted_indices = np.argmax(predictions, axis=1)
    row_labels        = le.inverse_transform(predicted_indices)
    predicted_labels_map.append(row_labels)

    # Top 5 predictions per pixel
    row_top5 = []
    for prob_vector in predictions:
        top5_indices = np.argsort(prob_vector)[::-1][:5]
        top5_labels  = le.inverse_transform(top5_indices)
        top5_probs   = prob_vector[top5_indices]
        row_top5.append(list(zip(top5_labels, top5_probs)))
    predicted_top5_map.append(row_top5)

predicted_labels_map = np.array(predicted_labels_map)   # (length, width)
predicted_top5_map   = np.array(predicted_top5_map, dtype=object)  # (length, width)

# Map mineral names to integers for colouring
unique_minerals = np.unique(predicted_labels_map)
mineral_to_int  = {m: i for i, m in enumerate(unique_minerals)}
label_map       = np.vectorize(mineral_to_int.get)(predicted_labels_map)

n_minerals = len(unique_minerals)
cmap       = plt.get_cmap('tab20', n_minerals)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(label_map, cmap=cmap, vmin=0, vmax=n_minerals - 1)

# Legend
legend_elements = [
    Patch(facecolor=cmap(i), label=mineral)
    for i, mineral in enumerate(unique_minerals)
]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
          loc='upper left', fontsize=8)
ax.set_axis_off()

# Annotation box that follows the cursor
annot = ax.annotate(
    "", xy=(0, 0), xytext=(10, 10),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="white", alpha=0.9),
    fontsize=8
)
annot.set_visible(False)

def on_hover(event):
    if event.inaxes == ax:
        # Convert cursor position to pixel indices
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        # Bounds check
        if 0 <= row < length and 0 <= col < width:
            top5 = predicted_top5_map[row, col]

            # Build annotation text
            lines = [f"Pixel ({row}, {col})", ""]
            for rank, (label, prob) in enumerate(top5, 1):
                lines.append(f"{rank}. {label}: {prob*100:.1f}%")

            annot.xy = (event.xdata, event.ydata)
            annot.set_text("\n".join(lines))
            annot.set_visible(True)
            fig.canvas.draw_idle()
    else:
        annot.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)

plt.tight_layout()
plt.savefig("mineral_map.png", dpi=150, bbox_inches='tight')
plt.show()