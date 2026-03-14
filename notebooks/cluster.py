from notebooks.raman_helper import *
import matplotlib.pyplot as plt

path = Path("~/Code/Data_SH/SB008").expanduser()
raman_data = Raman_Data(path, 10, 13)

cell_layer = raman_data.get_raw_hsi_cube()

preprocessing_pipeline = rp.preprocessing.Pipeline([
    rp.preprocessing.misc.Cropper(region=(150, 1200))
])

preprocessed_cell_layer = preprocessing_pipeline.apply(cell_layer)

kmeans = rp.analysis.cluster.KMeans(n_clusters=3)

clusters, cluster_centres = kmeans.apply(preprocessed_cell_layer)

rp.plot.spectra(
    cluster_centres, preprocessed_cell_layer.spectral_axis,
    plot_type="single stacked",
    label=[f"Cluster centre {i + 1}" for i in range(len(cluster_centres))]
    )
plt.show()

rp.plot.image(
    clusters, title=[f"Clusters {i + 1}" for i in range(len(clusters))],
    cbar=False
    )
plt.show()