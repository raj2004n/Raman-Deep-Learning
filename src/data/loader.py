import os
import numpy as np
import pandas as pd
import ramanspy as rp
from scipy.signal import convolve
from .grid import RamanGrid


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

def get_raw_hsi_cube(path, x, y, start=None, end=None):
    grid = RamanGrid(path, x, y)

    # read spectral axis from first file only, same across all so its fine
    files = grid.get_sorted_files()
    spectral_axis, _ = read_spectrum(files[0])

    # crop spectral axis and get start and end indexes of cropping region
    spectral_axis, start_idx, end_idx = crop_axis(spectral_axis, start, end)

    spectral_data = np.zeros((x, y, len(spectral_axis)))

    for file, row, col in grid.traverse():
        _, intensity = read_spectrum(file)
        # crop spectral_data to match spectral_axis, assign to spectral_data
        spectral_data[row, col] = intensity[start_idx:end_idx]
    
    return rp.SpectralImage(spectral_data, spectral_axis)

def get_area_under_hsi_cube(path, x, y, pipeline_id, rolling_window_width, start=None, end=None):
    grid = RamanGrid(path, x, y)
    pipeline = build_pipeline(pipeline_id)

    # cropper
    cropper = rp.preprocessing.misc.Cropper(region=(start, end))

    # read spectral axis from first file only, same across all so its fine
    files = grid.get_sorted_files()
    spectral_axis, _ = read_spectrum(files[0])

    # get cropped spectral axis
    temp_spectrum = rp.Spectrum(np.zeros(len(spectral_axis)), spectral_axis)
    temp_spectrum = cropper.apply(temp_spectrum)
    cropped_axis = temp_spectrum.spectral_axis

    # get number of indexes that correspond to rolling window width
    mean_step = np.mean(np.diff(cropped_axis))
    idx_step = max(1, int(rolling_window_width // mean_step))

    # trapezoidal kernel for area under curve using convolution
    kernel = np.ones(idx_step) * mean_step
    kernel[0] = 0.5 * mean_step
    kernel[-1] = 0.5 * mean_step

    n_bands = len(cropped_axis) - idx_step + 1 # accounting area
    area_cube = np.zeros((x, y, n_bands))
    spectrum_of_pixel = np.zeros((x * y + 1, len(cropped_axis)))
    pixel_map = np.zeros((x, y), dtype=int) # map of pixel positions

    pixel = 1 # initial pixel
    for file, row, col in grid.traverse():
        spectral_axis, spectral_data = read_spectrum(file)

        # crop and preprocess spectrum
        spectrum = rp.Spectrum(spectral_data, spectral_axis)
        spectrum = cropper.apply(spectrum)
        if pipeline is not None:
            spectrum = pipeline.apply(spectrum)

        spectral_data = spectrum.spectral_data # preprocessed now

        # store full preprocessed spectral data for plotting later
        spectrum_of_pixel[pixel] = spectral_data

        # convolve with kernet to get rolling window areas
        area_cube[row, col] = convolve(spectral_data, kernel, mode='valid')

        # store for plotting later
        pixel_map[row, col] = pixel
        pixel += 1
    
    return area_cube, spectrum_of_pixel, cropped_axis, idx_step, pixel_map