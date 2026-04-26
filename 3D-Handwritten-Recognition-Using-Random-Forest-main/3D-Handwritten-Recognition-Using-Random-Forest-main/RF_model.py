import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import pickle

class RandomForestFromScratch:
    """
    Random Forest model that trains and capable of predicting data labels

    Parameter: data used for training or testing, number of trees in the forest,
    number, Maximu depth for each tree, etc

    Return: A train model capable of predictin class labesl for a sample or set of samples
    
    """
    def __init__(self, n_estimators=100, max_depth=None, max_features="auto"):
        self.n_estimators = n_estimators  # Number of trees in the forest
        self.max_depth = max_depth        # Max depth for each tree
        self.max_features = max_features  # Max number of features to consider per split
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """ Create a bootstrap sample (random sample with replacement) """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _random_features(self, X):
        """ Randomly select a subset of features """
        if self.max_features == "auto":
            # Default to sqrt of the number of features
            max_features = int(np.sqrt(X.shape[1]))
        else:
            max_features = self.max_features
        
        features = np.random.choice(X.shape[1], size=max_features, replace=False)
        return features

    def fit(self, X, y):
        """ Train the random forest """
        for _ in range(self.n_estimators):
            # Step 1: Bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Step 2: Select random features for the decision tree
            features = self._random_features(X_sample)
            
            # Step 3: Train the decision tree with the subset of features
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=len(features))
            tree.fit(X_sample[:, features], y_sample)
            
            # Save the tree and the subset of features
            self.trees.append((tree, features))
            
    def predict(self, X):
        """ Predict using the random forest """
        # Get predictions from each tree
        tree_preds = np.array([tree.predict(X[:, features]) for tree, features in self.trees])
        
        tree_preds = tree_preds.T
        
        # Vote for the most common class in each row (sample)
        final_preds = [Counter(row).most_common(1)[0][0] for row in tree_preds]
        return np.array(final_preds)

    def predict_single_sample(self, sample):
        """
         Predicts the class of a single input sample using the trained RandomForestFromScratch model.
            
            Args:
            - An instance of RandomForestFromScratch (already trained).
            - sample: A 1D numpy array representing a single input sample.
            
            Returns:
            - Predicted class for the input sample.
        """
        # Reshape the single sample to 2D, as the model expects a batch of samples
        sample = sample.reshape(1, -1)
        
        # Use the predict method of the model
        prediction = self.predict(sample)
        
        # Return the single prediction (first element of the result)
        return prediction[0]
    



#Saving the model parameters
