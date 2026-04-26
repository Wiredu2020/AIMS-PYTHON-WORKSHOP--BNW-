import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

path = 'D:\Desktop\PRML_PROJECT\digits_3d'

def create_3d_dataset(path):
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
    
    return X, y

X,y = create_3d_dataset(path)

max_len = 222
X_padded = pad_sequences([x for x in X], maxlen=max_len, dtype='float32', padding='post')
X_flat = X_padded.reshape(1000, -1) 

# Assuming X_padded is already padded and y is the labels
X_flat = X_padded.reshape(1000, -1)  # Flatten the padded data
print("Shape of flattened X is", X_flat.shape)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.1, random_state=42)

# List of KNN configurations to try
knn_params = [
    {'n_neighbors': 3, 'weights': 'uniform', 'metric': 'euclidean'},
    {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean'},
    {'n_neighbors': 3, 'weights': 'distance', 'metric': 'euclidean'},
    {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'},
    {'n_neighbors': 7, 'weights': 'uniform', 'metric': 'manhattan'},
    {'n_neighbors': 7, 'weights': 'distance', 'metric': 'manhattan'}
]

# Iterate through each configuration
results = []
for params in knn_params:
    # Initialize the KNN classifier with given parameters
    knn = KNeighborsClassifier(**params)
    
    # Train the KNN model
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    # Evaluate the model's performance
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Save results
    results.append({
        'params': params,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })

    # Print results for this configuration
    print(f"Parameters: {params}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print('-' * 50)

# Summarize all results
print("\nSummary of Results:")
for result in results:
    print(f"Params: {result['params']} | Train Acc: {result['train_accuracy']:.4f} | Test Acc: {result['test_accuracy']:.4f}")
