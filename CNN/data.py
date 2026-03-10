import os
import numpy as np
import ramanspy as rp
from collections import defaultdict
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
import keras
from keras.utils import to_categorical

def get_data():
    path1 = os.path.expanduser('~/Code/Data_SH/unrated_oriented')
    path2 = os.path.expanduser('~/Code/Data_SH/unrated_unoriented')

    #spectra1, metadata1 = rp.datasets.rruff(path1, download=False)
    spectra2, metadata2 = rp.datasets.rruff(path2, download=False)

    #spectra  = spectra1 + spectra2
    #metadata = metadata1 + metadata2
    return spectra2, metadata2

def standardise_data(spectra_list, target_length, x_min=100, x_max=1800):
    standardised_data = []
    # new wavenumber axis the intensity will be interpolated to
    new_x = np.linspace(x_min, x_max, target_length)
    # get all standardised data
    for spectrum in spectra_list:
        x = spectrum.spectral_axis
        y = spectrum.spectral_data
        # interpolator
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
        # new interpolated intensities
        new_y = f(new_x)
        # clip the new_y to avoid intensity being below 0
        new_y = np.clip(new_y, 0, None)
        # normalise
        new_y /= (new_y.max() + 1e-8) # plus small value to avoid div by 0 error
        # append new y as a column as required by CNN model
        standardised_data.append(new_y.reshape(-1, 1))
    # returns shape (target_length, 1)
    return np.array(standardised_data)

def augment_shift(spectrum):
    intensity = spectrum[:, 0].copy() # again, the shape is (target_length, 1)

    # generate some random shift integer val between -5 and 5
    shift = np.random.randint(-5, 6)
    # shifts = 0 do nothing
    while shift == 0:
        shift = np.random.randint(-5, 6)

    # use numpy roll to shift for its quickness
    intensity = np.roll(intensity, shift)
    # np roll wraps, replace values to 0 that were wrapped around
    if shift > 0:
        intensity[:shift] = 0
    else:
        intensity[shift:] = 0

    # reshape to fit CNN model's req
    spectrum = intensity.reshape(-1, 1)
    return spectrum

def augment_noise(spectrum):
    intensity = spectrum[:, 0].copy()
    #noise = np.random.normal(0, 0.05 * np.abs(intensity))
    noise = np.random.normal(0, 0.05 * np.abs(intensity))

    # clip to stay bounded between 0 and inf
    intensity = np.clip(intensity + noise, 0, None)
    # normalise
    intensity /= (intensity.max() + 1e-8)
    # reshape to fit CNN model's req
    spectrum = intensity.reshape(-1, 1)
    return spectrum

def augment_linear_combinations(spectra_for_class, n_combinations):
    augmented = []
    n = len(spectra_for_class)
    
    if n < 2:
        return augmented # can't combine single spectrum
    
    for _ in range(n_combinations):
        # random coeff that sum up to one
        coeffs = np.random.dirichlet(np.ones(n))

        combined_intensity = sum(c * s[:, 0] for c, s in zip(coeffs, spectra_for_class))
        combined_intensity /= (combined_intensity.max() + 1e-8)
        augmented.append(combined_intensity.reshape(-1, 1))
    
    return augmented

def build_augmented_dataset(x_train, y_train_labels, n_combination):
    augmented_x = []
    augmented_y = []

    # collect the spectras grouped by classes
    # running hash map appraoch
    class_spectra = {}
    for spectrum, label in zip(x_train, y_train_labels):
        if label not in class_spectra:
            class_spectra[label] = []
        class_spectra[label].append(spectrum)

    for label, spectra_list in class_spectra.items():            
        # augment by linear combinations
        combos = augment_linear_combinations(spectra_list, n_combination)
        for combo in combos:
            augmented_x.append(combo)
            augmented_y.append(label)
    
    return np.array(augmented_x), np.array(augmented_y)

def leave_one_out_split(y_data):
    # since dictionary can not hold duplicate keys, use that to hold index
    class_indices = defaultdict(list)
    for i, label in enumerate(y_data):
        class_indices[label].append(i)
    
    train_index = []
    test_index = []

    for label, indices in class_indices.items():
        if len(indices) == 1:
            train_index.extend(indices)
        else:
            test_pick = np.random.choice(indices)
            test_index.append(test_pick)
            train_index.extend([i for i in indices if i != test_pick])
    
    return np.array(train_index), np.array(test_index)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, num_classes, batch_size, shuffle=True, augment=True):
        # hard coded
        self.dim = [913]
        self.n_channels = 1 # only intensity

        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()


    def __len__(self):
        'Number of batches per epoch'
        # this does not make the number of ephocs will depend augmented data
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch'
        # indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get one batch of data
        x_batch, y_batch = self.__data_generation(indexes)
        return x_batch, y_batch
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation1(self, indexes):
        'Generates data containing batch_size samples'
        
        if self.augment:
            # 4 versions per sample: original, shift, noise, shift+noise
            x_batch = np.empty((self.batch_size * 4, *self.dim, self.n_channels))
            y_batch = np.empty((self.batch_size * 4), dtype=int)
        else:
            x_batch = np.empty((self.batch_size, *self.dim, self.n_channels))
            y_batch = np.empty((self.batch_size), dtype=int)

        for i, index in enumerate(indexes):
            spectrum = self.x[index].copy()
            label    = self.y[index]

            if self.augment:
                shift_only      = augment_shift(spectrum)
                noise_only      = augment_noise(spectrum)
                shift_and_noise = augment_noise(augment_shift(spectrum))

                x_batch[i * 4]     = spectrum
                x_batch[i * 4 + 1] = shift_only
                x_batch[i * 4 + 2] = noise_only
                x_batch[i * 4 + 3] = shift_and_noise

                y_batch[i * 4]     = label
                y_batch[i * 4 + 1] = label
                y_batch[i * 4 + 2] = label
                y_batch[i * 4 + 3] = label
            else:
                x_batch[i] = spectrum
                y_batch[i] = label

        return x_batch, to_categorical(y_batch, num_classes=self.num_classes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x_batch = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_batch = np.empty((self.batch_size), dtype=int)

        for i, index in enumerate(indexes):
            spectrum = self.x[index].copy()

            # augment spectrum with shift then noise
            if self.augment:
                spectrum = augment_shift(spectrum)
                spectrum = augment_noise(spectrum)

            x_batch[i] = spectrum
            y_batch[i] = self.y[index]

        return x_batch, to_categorical(y_batch, num_classes=self.num_classes)