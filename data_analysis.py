from raman_helper import *
from pysptools import material_count
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import kneed

class Analysis:
    def __init__(self, hsi_cube):
        self.hsi_cube = hsi_cube
        self.all_ns = self.get_predicted_ns()
        self.predicted_n = max(self.all_ns)
        self.confidence = self.determine_confidence()
        
    def n_by_pca(self):
        # get matrix containing spectral data
        X = self.hsi_cube.spectral_data

        # prepare data to shape (rows * cols, band_lenght), and scale
        rows, cols, b = X.shape
        X = X.reshape(rows * cols, b).astype(np.float64)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # principal components that capture 95% variance
        pca_85 = PCA(n_components=0.85)
        pca_85.fit_transform(X)
        n_85 = pca_85.n_components_

        # principal components by knee method
        pca_elbow = PCA(n_components=10)
        pca_elbow.fit_transform(X)
        ev = pca_elbow.explained_variance_
        x = np.arange(1, 11)
        kn = kneed.knee_locator.KneeLocator(x, ev, curve='convex', direction='decreasing')
        n_elbow = kn.elbow if kn.elbow else None

        """
        # Kaiser's method: Only keep those whose eigenvalues greater than 1.
        # In practice, principal components with eigenvales like 0.95 may still
        # contain significant variance. Hence, this method is not the only one being used.
        pca_kaiser = PCA()
        pca_kaiser.fit_transform(X)
        ev = pca_kaiser.explained_variance_
        # pick out principal components with eigenvalue >= 1
        n_kaiser = np.sum(ev >= 1.0)
        """
        return n_85, n_elbow

    #TODO: Clean up raman_helper.py, fix naming convention
    def n_by_vd(self):
        # get spectral matrix
        X = self.hsi_cube.spectral_data
        # sclae spectral matrix
        X = (X - X.min()) / (X.max() - X.min())

        hfcvd = material_count.HfcVd()
        n_vd = hfcvd.count(X, far=[1e-5], noise_whitening=True)
        
        return n_vd[0]
    
    def get_predicted_ns(self):
        n_85, n_elbow = self.n_by_pca()
        n_vd = self.n_by_vd()
        return [n_85, n_elbow, n_vd]

    def determine_confidence(self):
        """
        Simple method to determine the confidence by finding the difference in predictions.

        Possible confidence values:
        - 2 three predictions match
        - 1 two predictions match
        - 0 no prediction match
        """
        p1, p2, p3 = self.all_ns

        if p1 == p2 and p2 == p3: # all predictions agreed
            return 'High'
        
        elif p1 == p2 or p1 == p3 or p2 == p3: # two predictions agreed
            return 'Good'
        
        #TODO: This should at least account difference
        else: # none agreed
            return 'None'

def main():
    path = Path("~/Code/Data_SH/SB008").expanduser()
    raman_data = Raman_Data(path, 10, 13)

    hsi_cube = raman_data.get_raw_slices()

    data_analysis = Analysis(hsi_cube)
    p1, p2, p3 = data_analysis.all_ns
    confidence = data_analysis.confidence
    print(f"Prediction: {p1, p2, p3}. Final Prediction: {data_analysis.predicted_n} with {data_analysis.confidence} Confidence.")

if __name__ == "__main__":
    main()