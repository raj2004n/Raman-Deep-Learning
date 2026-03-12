import numpy as np
import keras
import pickle
from src.data.loader import get_raw_hsi_cube
from src.cnn.cnn_preprocessing import standardise_data

input_size = 1024

def load_artifacts(model_path, encoder_path, wavenumber_path):
    model = keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)
    wavenumber_range = np.load(wavenumber_path)
    return model, le, wavenumber_range

def predict(path, x, y):
    model, le, wavenumber_range = load_artifacts(
        "artifacts/models/raman_cnn_model_poor.keras",
        "artifacts/encoders/label_encoder_poor.pkl",
        "artifacts/metadata/wavenumber_range_poor.npy"
    )

    x_min, x_max = wavenumber_range[0], wavenumber_range[1]

    hsi_cube = get_raw_hsi_cube(path, x, y)
    length, _ = hsi_cube.shape

    predicted_labels_map = []
    predicted_top5_map = []

    for l in range(length):
        # standardise each row of pixels for prediction
        spectra_list = list(hsi_cube[l])
        x_new = standardise_data(spectra_list, target_length=input_size, x_min=x_min, x_max=x_max)
        predictions = model.predict(x_new, verbose=0)  # (width, num_classes)

        # top prediction per pixel
        predicted_indices = np.argmax(predictions, axis=1)
        row_labels = le.inverse_transform(predicted_indices)
        predicted_labels_map.append(row_labels)

        # top 5 predictions per pixel
        row_top5 = []
        for prob_vector in predictions:
            top5_indices = np.argsort(prob_vector)[::-1][:5]
            top5_labels = le.inverse_transform(top5_indices)
            top5_probs = prob_vector[top5_indices]
            row_top5.append(list(zip(top5_labels, top5_probs)))
        predicted_top5_map.append(row_top5)

    predicted_labels_map = np.array(predicted_labels_map)
    predicted_top5_map = np.array(predicted_top5_map, dtype=object)

    return predicted_labels_map, predicted_top5_map