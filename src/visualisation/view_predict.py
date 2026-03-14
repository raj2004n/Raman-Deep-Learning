import numpy as np
import matplotlib.pyplot as plt
from src.cnn.predict import predict
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

def show_prediction_map(path, x, y, confidence_threshold=0.80, save_path=None):
    predicted_labels_map, predicted_top5_map = predict(path, x, y)
    
    length, width = predicted_labels_map.shape

    # build confidence map from top-1 probability
    confidence_map = np.array([
        [predicted_top5_map[r, c][0][1] for c in range(width)]
        for r in range(length)
    ])

    # mask low confidence pixels as 'Unknown'
    masked_labels = predicted_labels_map.copy()
    masked_labels[confidence_map < confidence_threshold] = 'Unknown'

    # map mineral names to integers for colouring
    unique_minerals = sorted([m for m in np.unique(masked_labels) if m != 'Unknown'])
    if 'Unknown' in np.unique(masked_labels):
        unique_minerals = unique_minerals + ['Unknown']

    mineral_to_int = {m: i for i, m in enumerate(unique_minerals)}
    label_map = np.vectorize(mineral_to_int.get)(masked_labels)

    n_minerals = len(unique_minerals)
    colors = [plt.get_cmap('tab20')(i / max(n_minerals - 1, 1)) for i in range(n_minerals)]
    if 'Unknown' in unique_minerals:
        colors[-1] = (0.6, 0.6, 0.6, 1.0)

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(label_map, cmap=cmap, vmin=0, vmax=n_minerals - 1)
    ax.set_axis_off()
    ax.set_title(f"Mineral Prediction Map (confidence threshold: {confidence_threshold*100:.0f}%)")

    # legend
    legend_elements = [
        Patch(facecolor=colors[i], label=mineral)
        for i, mineral in enumerate(unique_minerals)
    ]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8,
        title="Minerals",
        title_fontsize=9,
        framealpha=0.9
    )

    # hover annotation
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        fontsize=8
    )
    annot.set_visible(False)

    def on_hover(event):
        if event.inaxes == ax:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            if 0 <= row < length and 0 <= col < width:
                conf = confidence_map[row, col]
                if conf < confidence_threshold:
                    lines = [f"Pixel ({row}, {col})", "", "Unknown", f"Confidence: {conf*100:.1f}% (below threshold)"]
                else:
                    top5 = predicted_top5_map[row, col]
                    lines = [f"Pixel ({row}, {col})", ""]
                    for rank, (label, prob) in enumerate(top5[:5], 1):
                        lines.append(f"{rank}. {label}: {prob*100:.1f}%")
                annot.xy = (event.xdata, event.ydata)
                annot.set_text("\n".join(lines))
                annot.set_visible(True)
            else:
                annot.set_visible(False)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()