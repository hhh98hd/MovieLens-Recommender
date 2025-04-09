import numpy as np

from data.dataset import MovieLensDataset
from recommender import Recommender

if __name__ == "__main__":
    dataset = MovieLensDataset("data/ml-100k/")
    train_pivot, val_df, test_df = dataset.split_dataset()
 
    recommender = Recommender(method=Recommender.SVD_NAIVE)
    recommender.load_train_dataset(train_pivot)
    recommender.load_test_dataset(val_df)
        
    ks = np.arange(1, 963, 1)
    
    for k in ks:
        print(str(k))
        recommender.fit(k=k)
        print(recommender.evaluate())
        print("\n")
    