import numpy as np
import pandas as pd

class Recommender:
    SVD_NAIVE = 0
    
    def __init__(self, method=SVD_NAIVE):
        self._method = method
        self._pred = None
        
    def load_train_dataset(self, dataset: pd.DataFrame):
        """Load the dataset (the pivot) into the recommender model.

        Args:
            dataset (pd.DataFrame): The pivot user_id - ratings pivot dataframe.
        """
        self._train_dataset = dataset
        
    def load_test_dataset(self, dataset: pd.DataFrame):
        """Load the test dataset into the recommender model.

        Args:
            dataset (pd.DataFrame): The test dataset.
        """
        self._test_dataset = dataset
        
    def fit(self, k : int):
        """Fit the recommender model.
        
        Args:
            k (int): The number of latent features to use in the SVD.
        """
        
        if self._method == self.SVD_NAIVE:
            self._fit_naive(k)
        else:
            raise ValueError("Unknown method")
        
    def _fit_naive(self, k : int):
        """Fit the recommender model using naive SVD.

        Args:
            k (int): The number of latent features to use in the SVD.
        """
        
        ratings = self._train_dataset.fillna(self._train_dataset.mean(axis=0)).values
        
        u, e, v = np.linalg.svd(ratings, full_matrices=False)
        # Retain top-k latent features
        u = u[:, :k]
        e = e[:k]
        v = v[:k, :]
        
        self._pred = np.dot(np.dot(u, np.diag(e)), v)
        self._pred[self._pred < 0] = 0
    
    def predict(self, user_id : int, item_id : int) -> float:
        """Predict the rating for a given user and item.

        Args:
            user_id (int): The user ID.
            item_id (int): The item ID.

        Returns:
            float: The predicted rating.
        """
        
        if self._pred is None:
            raise ValueError("Model not fitted yet")
        
        return self._pred[user_id - 1, item_id - 1]   
        
    def evaluate(self) -> float:
        """Evaluate the recommender model using RSME metric

        Returns:
            float: The RMSE score.
        """
        total_error = 0
        count = 0
        
        for user_id, item_id, rating, _ in self._test_dataset.itertuples(index=False):
            pred_rating = self.predict(user_id, item_id)
            error = rating - pred_rating
            total_error += error ** 2
            count += 1
            
        return np.sqrt(total_error / count)
    