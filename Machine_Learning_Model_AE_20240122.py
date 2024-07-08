# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:38:21 2024

@author: Hugh Littlehailes

Machine learning attempt characterisation of AE data
"""

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC

t_size = 1026
f_size = 1024
Zxx_size = 1
batch_size = 2
num_classes = 1
num_epochs = 2

# Load data from the HDF5 file
#with h5py.File('stft_results_20240206.h5', 'r') as hf:
#    spectrogram_group = hf['spectrogram_data_20gmin_Layer1_Input0']
#    
#    time_data = spectrogram_group['time'][:]
#    frequency_data = spectrogram_group['frequency'][:]
#    Zxx_data = spectrogram_group['Zxx'][:]
#
#with h5py.File('stft_results_20240206_C1Test.h5', 'r') as hf_test:
#    spectrogram_group_test = hf_test['spectrogram_data_15gmin_Layer2_Input0']
#    
#    time_data_test = spectrogram_group_test['time'][:]
#    frequency_data_test = spectrogram_group_test['frequency'][:]
#    Zxx_data_test = spectrogram_group_test['Zxx'][:]

def load_stft_data(file_path, max_length=None):
    stft_data = []
    labels = []

    with h5py.File(file_path, 'r') as hdf5_file:
        for data_name, group in hdf5_file.items():
            # Extract STFT data and label
            time = np.array(group['time'])
            frequency = np.array(group['frequency'])
            intensity = np.array(group['intensity'])
            label = group.attrs['label']
            #print("The shape of the time array is:", time.shape)
            #print("The shape of the frequency array is:", frequency.shape)
            #print("The shape of the intensity array is:",intensity.shape)
            # Check if the length of time matches other arrays
            min_length = min(len(time), len(frequency), len(intensity))
            # Set the maximum length for padding
            if max_length is None:
                max_length = min_length

            ## Pad and preprocess time, frequency, and intensity arrays separately
            time_processed = TimeSeriesScalerMinMax().fit_transform(time[:min_length].reshape(-1, 1, 1)).reshape(-1, max_length, 1)
            frequency_processed = TimeSeriesScalerMinMax().fit_transform(frequency[:min_length].reshape(-1, 1, 1)).reshape(-1, max_length, 1)
            intensity_processed = TimeSeriesScalerMinMax().fit_transform(intensity[:min_length].reshape(-1, 1, 1)).reshape(-1, max_length, 1)
            
            print("Shapes before concatenation:")
            print("Time processed:", time_processed.shape)
            print("Frequency processed:", frequency_processed.shape)
            print("Intensity processed:", intensity_processed.shape)
            
            # Reshape time and frequency arrays to have an additional dimension
            time_reshaped = time[:, np.newaxis]  # shape: (1024, 1)
            frequency_reshaped = frequency[:, np.newaxis]  # shape: (1024, 1)
            
            # Concatenate time, frequency, and intensity arrays along the correct axis
            ts_data = np.concatenate((time_processed, frequency_processed, intensity_processed), axis=0)
            print("Shape after concatenation:", ts_data.shape)
            stft_data.append(ts_data)

            labels.append(label)

    return np.array(stft_data), np.array(labels)

# Assuming Zxx_data is normalized (Zxx_data = Zxx_data / np.max(Zxx_data))
# Reshape the arrays to have the same dimensions along the stacking axis
#time_data = np.reshape(time_data, (time_data.shape[0], 1))
#frequency_data = np.reshape(frequency_data, (frequency_data.shape[0], 1))
#Zxx_data = np.reshape(Zxx_data, (Zxx_data.shape[0], 1))


X_train, y_train = load_stft_data('stft_results_20240206.h5')
X_test, y_test = load_stft_data('stft_results_20240206_C1Test.h5')

print(y_train.shape)
# Combine time, frequency, and Zxx_data into a single 3D array
#spectrogram_data = np.stack((time_data, frequency_data, Zxx_data), axis=-1)
#spectrogram_data_test = np.stack((time_data_test, frequency_data_test, Zxx_data_test), axis=-1)

# Assuming label_data is your target labels
# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(spectrogram_data, label_data, test_size=0.2, random_state=42)
#_train, y_train = train_test_split(spectrogram_data, "15gmin_Layer1")#, test_size=0.2, random_state=42)
#X_test, y_test = train_test_split(spectrogram_data_test, "15gmin_layer2")#, test_size=0.2, random_state=42)

#X_train = np.array(spectrogram_data)
#X_test = np.array(spectrogram_data_test)

#y_train = ("15gmin_layer1")
#y_test = ("15gmin_layer2")

# Train machine learning model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Assuming num_classes is the number of classes in your classification task
#y_train_one_hot = to_categorical(y_train, num_classes)

#X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
#y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)  # Assuming y_train_one_hot is already converted


# Define the model
#model = models.Sequential()



# Add layers to the model
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(t_size, f_size, Zxx_size)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Flatten())
#model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(num_classes, activation='softmax'))  # Adjust num_classes based on your task

# Compile the model
#model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])#
#
# Train the model
#model.fit(X_train_tensor, y_train, epochs=num_epochs, batch_size=batch_size)#, validation_split=0.1)

# Evaluate the model on the test set
#test_loss, test_acc = model.evaluate(X_test, y_test)
#print(f'Test accuracy: {test_acc}')