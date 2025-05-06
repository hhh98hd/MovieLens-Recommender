import pandas as pd

from data.dataset import MovieLensDataset
from recommender import Recommender

def tune_k(recommender : Recommender) -> int:
    best_k = -1
    best_rmse = float('inf')
    
    ks = [1, 2, 5, 7, 10, 20, 50, 70, 100, 150, 175, 200, 500]
    for k in ks:
        recommender.fit(k)
        rmse = recommender.evaluate()
        
        print(f"RMSE for k={k}: {rmse}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
    
    return best_k

if __name__ == "__main__":
    dataset = MovieLensDataset("data/ml-20m/")
    
    print(dataset._movie_df)
    
    # train_df, val_df, test_df = dataset.split_dataset()
    
    # train_df = train_df.drop(columns=['timestamp'])
    # val_df = val_df.drop(columns=['timestamp'])
    # test_df = test_df.drop(columns=['timestamp'])
    
    # recommender = Recommender(  user_count=dataset.user_count, movie_count=dataset.movie_count,
    #                             user_ids=dataset.user_ids, movie_ids=dataset.movie_ids,
    #                             userid_to_idx=dataset.userid_to_idx, movieid_to_idx=dataset.movieid_to_idx,
    #                             method=Recommender.SVD_NAIVE )
    # recommender.load_train_dataset(train_df)
    # recommender.load_test_dataset(val_df)
    
    
    # best_k = tune_k(recommender)
    
    # # Merge the train and validation datasets
    # train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    # recommender.load_train_dataset(train_val_df)
    # recommender.load_test_dataset(test_df)
    
    # recommender.fit(best_k)
    
    # rmse = recommender.evaluate()
    # print(f"RMSE: {rmse}")