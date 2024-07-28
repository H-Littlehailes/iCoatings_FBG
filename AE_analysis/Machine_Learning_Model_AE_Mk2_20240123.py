# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:13:46 2024

@author: Hugh Littlehailes
"""

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

t_size = 3908
f_size = 1024
Zxx_size = 1024
batch_size = 2
num_classes = 2
num_epochs = 2

# Load data from the HDF5 file
with h5py.File('Coating2_Stripe1_15gmin_Layer1_Input0_Norm_Data2.h5', 'r') as hf:
    spectrogram_group = hf['spectrogram_data_15gmin_Layer1_Input0']
    time_data = spectrogram_group['time'][:]
    frequency_data = spectrogram_group['frequency'][:]
    Zxx_data = spectrogram_group['Zxx'][:]

with h5py.File('Coating2_Stripe1_15gmin_Layer2_Input0_Norm_Data2.h5', 'r') as hf_test:
    spectrogram_group_test = hf_test['spectrogram_data_15gmin_Layer2_Input0']
    time_data_test = spectrogram_group_test['time'][:]
    frequency_data_test = spectrogram_group_test['frequency'][:]
    Zxx_data_test = spectrogram_group_test['Zxx'][:]

# Reshape time_data and frequency_data to have the same dimensions
time_data_reshaped = np.reshape(time_data, (1024, 3))
frequency_data_reshaped = np.reshape(frequency_data, (1, frequency_data.shape[0]))

# Stack the arrays along the last axis
X_train = np.concatenate((time_data_reshaped, frequency_data_reshaped, Zxx_data), axis=-1)


# Stack the spectrogram data
#X_train = np.stack((time_data, frequency_data, Zxx_data), axis=-1)
X_test = np.stack((time_data_test, frequency_data_test, Zxx_data_test), axis=-1)

# Labels need to be integers, not strings
y_train = np.array([0] * len(X_train))  # Example: using 0 for "layer1"
y_test = np.array([1] * len(X_test))   # Example: using 1 for "layer2"

# Assuming num_classes is the number of classes in your classification task
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train_one_hot, dtype=tf.float32)

# Define the model
model = models.Sequential()

# Add layers to the model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(t_size, f_size, Zxx_size)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))  # Adjust num_classes based on your task

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tensor, y_train_tensor, epochs=num_epochs, batch_size=batch_size)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
print(f'Test accuracy: {test_acc}')
