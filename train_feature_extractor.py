import torch
from torch.utils.data import Dataset, random_split

from feature_extractor.feature_extractors import GenreAE, HCAE, train_feature_extractor
from data.dataset import MovieLensDataset

class MovieDataset(Dataset):
    def __init__(self, data_df):
        super().__init__()
        self._data = data_df.drop(columns=["movieId", "title"]).to_numpy().astype("float32")
        # Normalize the year column to be between 0 and 1
        self._data[:, -1] = (self._data[:, -1] - self._data[:, -1].min()) / (self._data[:, -1].max() - self._data[:, -1].min())
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]

movie_dataset = MovieDataset(MovieLensDataset("./data/ml-20m/")._movie_df)
dataset_size = len(movie_dataset)
val_size = int(0.1 * dataset_size)
test_size = int(0.1 * dataset_size)
train_size = dataset_size - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(
    movie_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(8535)
)

train_val_dataset = train_dataset + val_dataset

dims = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
best_dim = 0
best_mse = float("inf")

file = open("./feature_extractor/logs/HCAE.txt", "w+")

for dim in dims:
    model = HCAE(input_dim=movie_dataset[0].shape[0], latent_dim=dim)
    model.cuda()
    print(f"Training with latent dimension: {dim}")
    
    mse, avg_loss = train_feature_extractor(model, train_dataset, val_dataset, 
                                            latent_dim=dim, 
                                            epochs=200, 
                                            batch_size=256, 
                                            learning_rate=0.001)
    file.write(f"Latent dimension: {dim}, MSE: {mse}, Avg. Loss: {avg_loss}\n")
    
    if mse < best_mse:
        best_mse = mse
        best_dim = dim
        
print(f"Best latent dimension: {best_dim} with MSE: {best_mse}")
file.write(f"Best latent dimension: {best_dim} with MSE: {best_mse}\n")
file.close()

best_dim = 13
model = HCAE(input_dim=train_val_dataset[0].shape[0], latent_dim=best_dim).cuda()

mse, avg_loss = train_feature_extractor(model, train_val_dataset, test_dataset, latent_dim=best_dim)
