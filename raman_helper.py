from pathlib import Path 
import re
import numpy as np
import pandas as pd
import ramanspy as rp
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve

class Raman_Data:
    def __init__(self, path, x, y):
        self.path = path
        self.x = x
        self.y = y

    def get_file_number(self, path):
        """
        Returns the number of the corresponding file.

        eg. ".../FullGrid[1](-18000, -320, -14675)" -> 1

        If no name found, then returns None to signal 
        """
        match = re.search(r'\[(\d+)\]', path.name)
        return int(match.group(1)) if match else None

    def get_files(self):
        path = Path(self.path) # store as Path object for easier manipulation
        files = list(path.glob('*.txt')) # extract .txt files and store as list

        # if no files found
        if not files:
            print("No .txt files found in that directory.")
            return 
        
        # sort the files in the order of their file number
        files.sort(key=self.get_file_number)

        return files
    
    def step_grid(self, cur_x, cur_y, step):
        """
        Method to dictate current position traversing grid.
        
        :param cur_x: Current x position.
        :param cur_y: Current y position.
        :param step: Current step direction. +1 is right, -1 is left.
        """
 
        if cur_y == self.y - 1 and step != -1: # on right boundary
            cur_x -= 1 # step up
            step *= -1 # flip step direction
        elif cur_y == 0 and step != 1: # on left boundary
            cur_x -= 1 # step up
            step *= -1 # flip step direction
        else: # not on boundary
            cur_y += step # step 

        return cur_x, cur_y, step
    
    def get_area(self, pipeline):
        """
        Method to get the area under the whole intensity curve.
  
        :param pipeline: ...
        """

        # get files
        files = self.get_files()

        if pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()
        
        # list to hold area under curve
        integrals = np.zeros(shape=(self.x, self.y))

        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)

        for file in files:
            # read the intensity column and store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object and preprocess spectra
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
            #raman_spectra = pipeline.apply(raman_spectra)

            # get area under whole curve
            integral = np.trapezoid(intensity_arr, raman_shifts)

            # assign integral to its grid position
            integrals[cur_x, cur_y] = integral

            # step
            cur_x, cur_y, step = self.step_grid(cur_x, cur_y, step)

        return integrals, raman_shifts

    def get_area_regions(self, pipeline, regions):
        """
        Method to find the area under the curve by regions.
        
        Returns:
        area_by_region - integral of region, stored as: list, shape=(x, y, len(spectra)) 
        shift_by_region - raman shift stored as dictionary: {region : spectra of region} (same for all pixels)
        spectra_by_region - preprocessed spectra of region, stored as: { pixel, region : spectra of pixel and region}
        original_spectra_by_region - original spectra of region, stored as: { pixel, region : spectra of pixel and region}

        original_spectra_by_region was added to compare with preprocessed data

        :param pipeline: pipeline to use. Currently using predefined pipelines from RamanSPy library.
        :param regions: Number of regions to split the spectra into.
        """
        # get files
        files = self.get_files()

        if pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()
        
        div = len(raman_shifts) // regions
        area_by_region = np.zeros(shape=(self.x, self.y, regions))

        # key: pixel, region. value: spectra of that region for that pixel
        spectra_by_region = {}
        original_spectra_by_region = {}
        
        # key: region. value: spectra of that region
        shift_by_region = {}

        # store raman shift by region
        i, j, region = 0, div - 1, 0
       
        while i < len(raman_shifts):
            shift_by_region[region] = [f"{raman_shifts[i]} - {raman_shifts[j]}", raman_shifts[i:j]]
            region += 1
            i += div
            j += div
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
        pixel = 1
        for file in files:
            # read the intensity column and store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object and preprocess spectra
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
            raman_spectra = pipeline.apply(raman_spectra)

            # process data
            processed_data = raman_spectra.spectral_data

            # pointers for splitting
            i, j, region = 0, div - 1, 0

            # store data by regions
            while i < len(raman_shifts):
                # Note that does eg. 0 133 and 134 267, so relation between 133 and 134 are is missed
                area = np.trapezoid(processed_data[i:j], raman_shifts[i:j])
                # store area by region
                area_by_region[cur_x, cur_y, region] = area
                # store spectras
                spectra_by_region[pixel, region] = processed_data[i:j]
                original_spectra_by_region[pixel, region] = intensity_arr[i:j] # store original for comparison
                # update pointers
                region += 1
                i += div
                j += div

            # step
            cur_x, cur_y, step = self.step_grid(cur_x, cur_y, step)
            pixel += 1

        return area_by_region, shift_by_region, spectra_by_region, original_spectra_by_region

    def get_area_range(self, pipeline, rolling_window, spectra_start, spectra_end):
        """
        Method to find the area under the curve by regions.
        
        Returns:
        area_by_region - integral of region, stored as: list, shape=(x, y, len(spectra)) 
        shift_by_region - raman shift stored as dictionary: {region : spectra of region} (same for all pixels)
        spectra_by_region - preprocessed spectra of region, stored as: { pixel, region : spectra of pixel and region}
        original_spectra_by_region - original spectra of region, stored as: { pixel, region : spectra of pixel and region}

        original_spectra_by_region was added to compare with preprocessed data

        :param pipeline: pipeline to use. Currently using predefined pipelines from RamanSPy library.
        :param regions:
        """
        # get files
        files = self.get_files()
        if pipeline == 0:
            pipeline = None
        elif pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)

        # cropper
        cropper = rp.preprocessing.misc.Cropper(region=(spectra_start, spectra_end))

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()
        
        temp_intensity_arr = pd.read_csv(files[0], sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].values
        temp_spectra = rp.Spectrum(temp_intensity_arr, raman_shifts)
        temp_spectra = cropper.apply(temp_spectra)

        # cropped raman shfit
        cropped_raman_shifts = temp_spectra.spectral_axis

        # steps between adjacent numbers and overall mean
        steps = np.diff(cropped_raman_shifts)
        mean_step = np.mean(steps) # one mean step corresponds to one index step

        # rolling window is greater than the mean step
        if rolling_window > mean_step:
            idx_step = int(rolling_window // mean_step)
        else:# range is less than the mean step
            idx_step = 1 # min step

        kernel = np.ones(idx_step)
        kernel[0] = 0.5
        kernel[-1] = 0.5
        kernel = kernel * mean_step

        area_by_region = np.zeros(shape=(self.x, self.y, len(cropped_raman_shifts) - idx_step + 1))
        spectra_by_pixel = np.zeros(shape=(self.x * self.y + 1, len(cropped_raman_shifts)))
        pixel_map = np.zeros(shape=(self.x, self.y), dtype=int)
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
        pixel = 1

        for file in files:
            # read the intensity column and store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].values
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts) # make raman spectra object
            raman_spectra = cropper.apply(raman_spectra) # crop the spectra

            if pipeline is not None:
                raman_spectra = pipeline.apply(raman_spectra) # apply preprocessing
            preprocessed_data = raman_spectra.spectral_data # extract preprocessed data

            areas = convolve(preprocessed_data, kernel, mode='valid')

            area_by_region[cur_x, cur_y, :] = areas

            # add spectra to pixel
            spectra_by_pixel[pixel] = preprocessed_data

            pixel_map[cur_x, cur_y] = pixel

            # step
            cur_x, cur_y, step = self.step_grid(cur_x, cur_y, step)
            # move to next pixel
            pixel += 1
        
        return area_by_region, spectra_by_pixel, cropped_raman_shifts, idx_step, pixel_map

    def get_raw_slices(self):
        """
        Method to get all the slices of the Raman Image. 
        
        By 'slice' I mean the Raman image for that Raman Shift value.
        
        :param pipeline: ...
        """
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()

        spectral_data = np.zeros(shape=(self.x, self.y, len(raman_shifts))) 
        spectral_axis =  np.array(raman_shifts)
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)

        for file in files:
            # read the intensity column, store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None,usecols=[1],)['intensity'].tolist()
            
            # assign integral to its grid position
            spectral_data[cur_x, cur_y] = intensity_arr

            # step
            cur_x, cur_y, step = self.step_grid(cur_x, cur_y, step)

        raman_slice = rp.SpectralImage(spectral_data, spectral_axis)

        return raman_slice
    
    def get_slice500(self, pipeline):
        """
        Method to get the first 500 slices of Raman Images. 
        Used to try out their trained neural network model, as it requires spectra length of 500.
        """
        # get files
        files = self.get_files()

        if pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()

        spectral_data = np.zeros(shape=(self.x, self.y, 500)) 
        spectral_axis =  np.array(raman_shifts[0:500])
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
  
        for file in files:
            # read the intensity column, store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None,usecols=[1],)['intensity'].tolist()
            
            # trim data
            intensity_arr = intensity_arr[0:500]

            # assign integral to its grid position
            spectral_data[cur_x, cur_y] = intensity_arr

            # step
            cur_x, cur_y, step = self.step_grid(cur_x, cur_y, step)

        raman_slice = rp.SpectralContainer(spectral_data, spectral_axis)
        raman_slice = pipeline.apply(raman_slice)

        # return slice
        return rp.SpectralContainer(spectral_data, spectral_axis)
    
    def get_all_spectra(self, pipeline):
        """
        Method to get the spectra for every pixel
        
        :param pipeline: ...
        """
        # get files
        files = self.get_files()

        if pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()

        # { pixel, spectra}
        spectra_pixel = {}
        raw_spectra_pixel = {}
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
        pixel = 1

        for file in files:
            # read the intensity column and store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object and preprocess spectra
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)

            raw_spectra_pixel[pixel] = raman_spectra

            raman_spectra = pipeline.apply(raman_spectra)

            spectra_pixel[pixel] = raman_spectra

            # step
            cur_x, cur_y, step = self.step_grid(cur_x, cur_y, step)
            # increment pixel
            pixel += 1

        return spectra_pixel, raw_spectra_pixel, raman_shifts