import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences



# directory where the Excel files are stored
directory = "digits_3d/training_data"

# List all files in the directory
excel_files = [f for f in os.listdir(directory) if f.endswith('.csv')]


def create_3d_dataset():
    path  = directory
    X = []  
    y = []  

    for file in os.listdir(path):
        if file.endswith('.csv'):  
            parts = file.split('_')
            label = int(parts[1])  
            y.append(label)
            
            file_path = os.path.join(path, file)
            data = pd.read_csv(file_path, header=None).to_numpy()
            X.append(data)
    
    y = np.array(y)
    
    X,y = processed_data(X,y)
    return X, y


def processed_data(X,y):

    #Finding maximum length of data
    max_len = max([x.shape[0] for x in X])
    #padding them to maximum length
    X_padded = pad_sequences([x for x in X], maxlen=max_len, dtype='float32', padding='post')


    #Reshaping data to (1000,222,3)
    X_flat = X_padded.reshape(1000, -1) 

    indices = np.arange(X_flat.shape[0])  # Generate an array of indices
    np.random.shuffle(indices)  # Shuffle the indices

    X_flat_shuffled = X_flat[indices]  # Apply the shuffled indices to X_flat
    y_shuffled = y[indices]
    return X_flat_shuffled, y_shuffled

def preprocess(x, target_shape=(222, 3)):
    
    # Get the current number of rows in x
    current_rows = x.shape[0]
    target_rows = target_shape[0]
    
    # Calculate how many rows to add or trim
    rows_to_add = target_rows - current_rows
    
    if rows_to_add > 0:
        # Pad the array with zeros
        padding = np.zeros((rows_to_add, x.shape[1]))
        padded_x = np.vstack([x, padding])  # Stack the original array with the padding
    elif rows_to_add < 0:
        # Trim the array if it has more rows than the target shape
        padded_x = x[:target_rows, :]
    else:
        # If the number of rows is exactly the target size, return the array as is
        padded_x = x

    flattened_x = padded_x.flatten()  # Flattens the 2D array to 1D
    
    return flattened_x