import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


from util import measure_execution_time, measure_memory

class Recommender:
    SVD_NAIVE = 0
    HCAE = 1
    
    def __init__(self, user_count : int, movie_count : int,
                 user_ids : pd.Series, movie_ids : pd.Series, 
                 userid_to_idx : dict, movieid_to_idx : dict, 
                 method=SVD_NAIVE):
        self.user_count = user_count
        self.movie_count = movie_count
        self._method = method
        
        self._user_ids = user_ids
        self._movie_ids = movie_ids
        
        self._userid_to_idx = userid_to_idx
        self._movieid_to_idx = movieid_to_idx
        
        if method == self.HCAE:
            pass
    
    def load_train_dataset(self, dataset: pd.DataFrame):
        """Load the dataset (the pivot) into the recommender model.

        Args:
            dataset (pd.DataFrame): The pivot user_id - ratings pivot dataframe.
        """
        self._train_dataset = dataset.copy()
        
        train_user_ids = self._train_dataset['userId'].apply(lambda x: self._userid_to_idx[x])
        train_movie_ids = self._train_dataset['movieId'].apply(lambda x: self._movieid_to_idx[x])
        
        R_sparse = csr_matrix((self._train_dataset['rating'], (train_user_ids, train_movie_ids)), 
                          shape=(self.user_count, self.movie_count))
        
        # The list of movies with no ratings -> Perhaps should be filled with the global mean rating?
        self._rating_means, global_rating_mean = self._compute_movie_rating_means(R_sparse)
        self._rating_means[self._rating_means == 0] = global_rating_mean
        
        self._R_centered = R_sparse.copy()
        for i in range(R_sparse.nnz):
            movie_idx = R_sparse.indices[i]
            self._R_centered.data[i] -= self._rating_means[movie_idx]
    
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
    
    
    @measure_memory
    @measure_execution_time
    def _fit_naive(self, k : int):
        """Fit the recommender model using naive SVD.

        Args:
            k (int): The number of latent features to use in the SVD.
        """
        
        u, e, v = svds(self._R_centered, k=k)
        self._u = u[:, ::-1]          # reorder the columns in U
        self._e = e[::-1]             # [σ_large, σ_small]
        self._v = v[::-1, :]          # reorder the rows in VT
        
    @measure_execution_time
    def _fit_hcae(self, k : int):
        pass
    
    def predict(self, user_id : int, movie_id : int) -> float:
        """Predict the rating for a given user and item.

        Args:
            user_id (int): The user ID.
            movie_id (int): The movie ID.

        Returns:
            float: The predicted rating. A real number between 0 and 5.
        """
        
        if self._u is None:
            raise ValueError("Model not fitted yet")
        
        user_idx = self._userid_to_idx[user_id]
        movie_idx = self._movieid_to_idx[movie_id]
        
        pred = np.dot(self._u[user_idx, :] * self._e, self._v[:, movie_idx]) + self._rating_means[movie_idx]
        
        return pred
            
    def evaluate(self) -> float:
        """Evaluate the recommender model using RSME metric

        Returns:
            float: The RMSE score.
        """
        total_error = 0
        count = 0
        
        for user_id, item_id, rating in self._test_dataset.itertuples(index=False):
            pred_rating = self.predict(user_id, item_id)
            error = rating - pred_rating
            total_error += error ** 2
            count += 1
            
        return np.sqrt(total_error / count)
    
    def _compute_movie_rating_means(self, matrix : csr_matrix) -> tuple[np.ndarray, float]:
        global_mean = matrix.sum() / matrix.count_nonzero()
        
        # Convert to CSC (Sparse Column fomat) for efficient column operations
        matrix_csc = matrix.copy().tocsc()
        
        rating_sums = np.array(matrix_csc.sum(axis=0)).flatten()
        rating_counts = np.diff(matrix_csc.indptr)
        
        rating_means = np.divide(rating_sums, rating_counts, out=np.zeros_like(rating_sums), where=rating_counts != 0)
        
        return rating_means, global_mean
    