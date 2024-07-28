import h5py
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict

# Function to load data from HDF5 file
def load_data(file_path, max_size=1024):
    stft_data = []
    labels = []

    with h5py.File(file_path, 'r') as hdf5_file:
        for data_name, group in hdf5_file.items():
            # Extract STFT data and label
            time = np.array(group['time'])
            frequency = np.array(group['frequency'])
            intensity = np.array(group['intensity'])
            label = group.attrs['label']

            # Specify the desired size along both dimensions
            size_dim_0 = max_size
            size_dim_1 = max_size  # Assuming the inconsistency is in dimension 1

            # Pad or truncate each array individually
            time = pad_or_truncate(time, size_dim_0, size_dim_1, axis=1)
            frequency = pad_or_truncate(frequency, size_dim_0, size_dim_1, axis=1)
            intensity = pad_or_truncate(intensity, size_dim_0, size_dim_1, axis=1)

            # Reshape time and frequency to make them 2D
            time = time.reshape(-1, 1)
            frequency = frequency.reshape(-1, 1)

            # Stack time, frequency, and intensity arrays horizontally
            combined_data = np.hstack((time, frequency, intensity))

            # Append to the list of data and labels
            stft_data.append(combined_data)
            labels.append(label)

        # Concatenate the list of data arrays into a single array
        data = np.concatenate(stft_data, axis=0)

    return data, labels

def pad_or_truncate(array, target_size_dim_0, target_size_dim_1, axis=1):
    if array.ndim > 1:
        current_size_dim_1 = array.shape[axis]

        if current_size_dim_1 < target_size_dim_1:
            # Pad the array along dimension 1
            padding_dim_1 = target_size_dim_1 - current_size_dim_1
            pad_width = [(0, 0)] * array.ndim
            pad_width[axis] = (0, padding_dim_1)
            array = np.pad(array, pad_width, mode='constant', constant_values=0)
        elif current_size_dim_1 > target_size_dim_1:
            # Truncate the array along dimension 1
            array = array[:, :target_size_dim_1] if axis == 1 else array

        # Pad or truncate along dimension 0
        current_size_dim_0 = array.shape[0]
        if current_size_dim_0 < target_size_dim_0:
            # Pad the array along dimension 0
            padding_dim_0 = target_size_dim_0 - current_size_dim_0
            pad_width_dim_0 = (0, padding_dim_0)
            array = np.pad(array, pad_width=(pad_width_dim_0, (0, 0)), mode='constant', constant_values=0)
        elif current_size_dim_0 > target_size_dim_0:
            # Truncate the array along dimension 0
            array = array[:target_size_dim_0, :]

    return array


# Function to perform shapelet transform
def shapelet_transform(data, labels, shapelet_length=0.1):
    # Normalize the time series data
    data_normalized = TimeSeriesScalerMinMax().fit_transform(data)

    # Create a shapelet model
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=data.shape[0], 
                                                       n_classes=len(np.unique(labels)), 
                                                       l=0.1, 
                                                       r=1,
                                                       ts_sz=data.shape[1])  # Add ts_sz parameter
    model = ShapeletModel(n_shapelets_per_size=shapelet_sizes, optimizer="sgd", weight_regularizer=.01)

    # Fit the model
    model.fit(data_normalized, labels)

    # Transform the data using the learned shapelets
    transformed_data = model.transform(data_normalized)

    return transformed_data

# Example usage
file_path = 'stft_results_20240206.h5'  # Replace with the actual path to your HDF5 file
data, labels = load_data(file_path)

# Set the desired shapelet length (adjust as needed)
shapelet_length = 0.1

# Perform shapelet transform
transformed_data = shapelet_transform(data, labels, shapelet_length)

# Now 'transformed_data' contains the transformed time series data with shapelets
# You can use this transformed data for further analysis or classification
