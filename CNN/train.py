from model import *
from data import *
from evaluate import *
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import pickle
from collections import Counter

spectra_list, metadata = get_data()

y_data = [m['##NAMES'].split(',')[0].strip() for m in metadata]

train_index, test_index = leave_one_out_split(y_data)

# one hot encoding 
le = LabelEncoder()
le.fit(y_data)

classes = le.classes_
num_classes = len(classes)

y_data = np.array(y_data)

y_train_labels = y_data[train_index]
y_test_labels = y_data[test_index]

y_test = to_categorical(le.transform(y_test_labels), num_classes)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

spectra_train = [spectra_list[i] for i in train_index]
spectra_test = [spectra_list[i] for i in test_index]

# hard coded for now
x_min, x_max = 0, 1400

# standardise the training and testing data
x_train = standardise_data(spectra_train, target_length=913, x_min=x_min, x_max=x_max)
x_test = standardise_data(spectra_test, target_length=913, x_min=x_min, x_max=x_max)

# augment by summing possible linear combinations
x_combos, y_combos_labels = build_augmented_dataset(x_train, y_train_labels, n_combination=10)

# update training datasset to include augmented data
x_train = np.concatenate([x_train, x_combos], axis=0)
y_train_labels = np.concatenate([y_train_labels, y_combos_labels], axis=0)

model = CNN_model(num_classes)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True)

callback = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
# create a class array of ints, since classes are labelled as string 
classes_array = np.arange(num_classes)
y_train_integers = le.transform(y_train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_array, y=y_train_integers)
class_weight_dict = dict(enumerate(class_weights))

train_gen = DataGenerator(
    x_train, y_train_integers,
    num_classes,
    batch_size = 64,
    shuffle = True,
    augment = True
    )

val_gen = DataGenerator(
    x_test, le.transform(y_test_labels),
    num_classes,
    batch_size = 64,
    shuffle = False,
    augment = False
    )

history = model.fit(
    train_gen,
    epochs=500,
    callbacks=[callback],
    validation_data=val_gen,
    class_weight=class_weight_dict,
    )

from collections import Counter

train_counts = Counter(y_train_labels)
test_counts  = Counter(y_test_labels)
all_counts   = Counter(np.concatenate([y_train_labels, y_test_labels]))

single_sample_classes = sum(1 for c in all_counts if all_counts[c] == 1)
train_only_classes    = sum(1 for c in train_counts if c not in test_counts)

print(f"Training spectra:              {len(spectra_train)}")
print(f"Test spectra:                  {len(spectra_test)}")
print(f"Total classes:                 {num_classes}")
print(f"Classes with only 1 spectrum:  {single_sample_classes}")
print(f"Classes only in train set:     {train_only_classes}")
print(f"x_all size after augmentation: {len(x_train)}")
print(f"Steps per epoch:               {len(train_gen)}")
print(f"Stopped at epoch:              {len(history.history['loss'])}")

# Distribution breakdown
counts = list(all_counts.values())
print(f"\nSpectra per class distribution:")
print(f"  Min:    {min(counts)}")
print(f"  Max:    {max(counts)}")
print(f"  Mean:   {np.mean(counts):.1f}")
print(f"  Median: {np.median(counts):.1f}")
print(f"  Classes with 1 spectrum:  {sum(1 for c in counts if c == 1)}")
print(f"  Classes with 2 spectra:   {sum(1 for c in counts if c == 2)}")
print(f"  Classes with 3+ spectra:  {sum(1 for c in counts if c >= 3)}")


# save full model and wavenumber range
model.save("raman_cnn_model.keras")
model.save_weights("test_weights.weights.h5")
np.save("wavenumber_range.npy", np.array([x_min, x_max]))

show_results(history, model, x_test, y_test)