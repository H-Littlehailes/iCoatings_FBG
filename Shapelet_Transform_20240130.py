import h5py
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from tqdm import tqdm  # Import tqdm for the loading bar


# Function to load STFT data from the HDF5 file and preprocess time and frequency arrays
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
            #time_processed = TimeSeriesScalerMinMax().fit_transform(time[:min_length].reshape(-1, 1, 1)).reshape(-1, max_length, 1)
            #frequency_processed = TimeSeriesScalerMinMax().fit_transform(frequency[:min_length].reshape(-1, 1, 1)).reshape(-1, max_length, 1)
            #intensity_processed = TimeSeriesScalerMinMax().fit_transform(intensity[:min_length].reshape(-1, 1, 1)).reshape(-1, max_length, 1)
            
            
            # Reshape time and frequency arrays to have an additional dimension
            time_reshaped = time[:, np.newaxis]  # shape: (1024, 1)
            frequency_reshaped = frequency[:, np.newaxis]  # shape: (1024, 1)

            # Concatenate time, frequency, and intensity arrays along the correct axis
            ts_data = np.concatenate((time_reshaped, frequency_reshaped, intensity), axis=1)

            # Combine time, frequency, and intensity into a single time series
            #print("Shapes before concatenation:")
            #print("Time processed:", time_processed.shape)
            #print("Frequency processed:", frequency_processed.shape)
            #print("Intensity processed:", intensity_processed.shape)

            #ts_data = np.concatenate((time_processed, frequency_processed, intensity_processed), axis=2)
            #print("Shape after concatenation:", ts_data.shape)
            
            stft_data.append(ts_data)

            # Combine time, frequency, and intensity into a single time series
            #ts_data = np.concatenate((time_processed, frequency_processed, intensity_processed), axis=2)

            #stft_data.append(ts_data)
            labels.append(label)

    return np.array(stft_data), np.array(labels)

# Load STFT data from the HDF5 file and preprocess time and frequency arrays
hdf5_filename = 'stft_results_20240206_RedBR1.h5'
X, y = load_stft_data(hdf5_filename)

# Print the shape of the data before applying the shapelet transform
print("Shape of X before shapelet transform:", X.shape)
ts_sz = X.shape[1]
# Set up a shapelet transform model
shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X.shape[0], 
                                                       n_classes=len(np.unique(y)), 
                                                       l=0.1, #l: Represents min length of shapelets. Shapelets are local, time-series subsequences that are considered discriminative for classification tasks. The l parameter allows you to specify the minimum length of these subsequences. Shapelets shorter than l will not be considered during the shapelet search process.
                                                       r=1, #Represents the max length/range of shapelets. Similar to l, r allows you to define an upper limit on the length of shapelets. Shapelets longer than r will not be considered during the shapelet search process.
                                                       ts_sz=ts_sz)
model = ShapeletModel(n_shapelets_per_size=shapelet_sizes, 
                      optimizer="sgd", 
                      weight_regularizer=.01, 
                      max_iter=2)

# Flatten the time series data to 2D before fitting the model
X_flat = X.reshape((X.shape[0], -1, X.shape[-1]))

# Fit the model on the data with tqdm progress bar
for _ in tqdm(range(1000), desc="Fitting Model", unit="epoch"):  # Just a dummy loop for demonstration
    model.fit(X_flat, y)

# Print the shapelets discovered by the model
print("Shapelets:")
for size in shapelet_sizes.keys():
    print(f"Size {size}: {model.shapelets_[size]}")