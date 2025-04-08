from data.dataset import MovieLensDataset
from recommender import Recommender

if __name__ == "__main__":
    dataset = MovieLensDataset("data/ml-100k/")
    
    recommender = Recommender(method=Recommender.SVD_NAIVE)
    recommender.load_dataset(dataset)
    recommender.fit(k=20)

    