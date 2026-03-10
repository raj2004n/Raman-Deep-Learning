import os
from pathlib import Path 
import ramanspy as rp
import pandas as pd
import numpy as np

def load_CNN_data(path="~/Code/Data_SH/poor_unoriented"):
    """Loads Raman Spectra dataset from RRUFF for training the CNN model.

    Args:
        path (str, optional): File path for the dataset. 
            Defaults to "~/Code/Data_SH/poor_unoriented".

    Returns:
        SpectralContainer, List[dict]: RamanSPy spectral container containing spectra and metadata
    """
    path = os.path.expanduser(path)
    spectra, metadata = rp.datasets.rruff(path, download=False)
    return spectra, metadata

def read_spectrum(file):
    """Reads the spectrum from the passed in file.

    Args:
        file : _description_

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(file, sep='\t', names=['raman_shift', 'intensity'], header=None, usecols=[0, 1])    
    return df['raman_shift'].tolist(), df['intensity'].values

def build_pipeline(pipeline_id, normalisation_pixelwise=True, fingerprint=False):
    if pipeline_id == 0:
        return None
    elif pipeline_id == 1:
        return rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise, fingerprint)
    elif pipeline_id == 2:
        return rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise, fingerprint)
    elif pipeline_id == 3:
        return rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise, fingerprint)
    else:
        raise ValueError(f"Unknown pipeline id: {pipeline_id}")
    
def crop_axis(spectral_axis, start=None, end=None):
    spectral_axis = np.array(spectral_axis)
    start_idx = np.searchsorted(spectral_axis, start) if start is not None else 0
    end_idx = np.searchsorted(spectral_axis, end) if end is not None else len(spectral_axis)
    return spectral_axis[start_idx:end_idx], start_idx, end_idx