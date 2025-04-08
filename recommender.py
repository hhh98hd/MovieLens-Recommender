import numpy as np

from data.dataset import MovieLensDataset

class Recommender:
    SVD_NAIVE = 0
    
    def __init__(self, method=SVD_NAIVE):
        self._method = method
        self._pred = None
        
    def load_dataset(self, dataset: MovieLensDataset):
        """Load the dataset into the recommender model.

        Args:
            dataset (MovieLensDataset): The dataset to load.
        """
        self._dataset = dataset
        
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
        
        self._R = self._dataset.get_rating_pivot().fillna(0).values
        
        u, e, v = np.linalg.svd(self._R, full_matrices=False)
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
        
      
