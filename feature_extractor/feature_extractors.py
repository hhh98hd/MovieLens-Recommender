import torch
import torch.nn as nn

from tqdm import tqdm

class GenreAE(nn.Module):
    def __init__(self, input_dim : int, latent_dim : int):
        super(GenreAE, self).__init__()
        
        self._enc = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, latent_dim),
            nn.ReLU(),
        )
        
        self._dec = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Encoder
        compressed = self._enc(x)
        
        # Decoder
        reconstructed = self._dec(compressed)
        
        return compressed, reconstructed


class HCAE(nn.Module):
    def __init__(self, input_dim : int, latent_dim : int, filter_num : int = 4, kernel_size : int = 5):
        super(HCAE, self).__init__()
       
        #1 Half-Convolutional Encoder
        self._encoder = nn.Sequential(
            nn.Conv1d(1, filter_num, kernel_size=kernel_size, padding=0, stride=1),
            nn.MaxPool1d(kernel_size=filter_num, stride=filter_num),
            nn.Flatten(),
        )
        
        # The formula mentioned in the paper
        conv_out_size = input_dim - kernel_size + 1                       # Padding = 0, Stride = 1
        pooling_out_size = (conv_out_size - filter_num) // filter_num + 1 # Padding = 0, Stride = filter_num
        
        #2 Compression Layer
        self._compression = nn.Sequential(
            nn.Linear(filter_num * pooling_out_size, latent_dim),
            nn.ReLU(),
        )
        
        self._dropout = nn.Dropout(0.2)
        
        #3 Fully Connected Decoder
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        #1 Half-Convolutional Encoder
        x = self._encoder(x)
        
        #2 Compression Layer
        x = self._compression(x)
        
        compressed = self._dropout(x)
        
        #3 Fully Connected Decoder
        reconstructed = self._dropout(compressed)
        reconstructed = self._decoder(reconstructed)
        
        return compressed, reconstructed
    
def train_feature_extractor(model : nn.Module, train_dataset, val_dataset, latent_dim : int, epochs : int = 200, batch_size : int = 32, learning_rate : float = 0.001):
    """
    Train the feature extractor model on the given dataset.
    
    Args:
    - model: The feature extractor model to train.
    - dataset: The dataset to train on.
    - epochs: Number of epochs to train for.
    - batch_size: Size of each training batch.
    - learning_rate: Learning rate for the optimizer.
    """
        
    # Set the model to training mode
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    total_loss = 0.0
    best_mse = float('inf')
    
    print(f'{"Epoch":>5} {"Avg. Loss":>25} {"Reconstruction Error":>15}')
    
    for epoch in range(epochs):
        for x in tqdm(data_loader):
            x = x.cuda()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, reconstructed = model(x)
            
            loss = criterion(reconstructed, x)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        error = evaluate_feature_extractor(model, val_dataset, batch_size)
    
        if error < best_mse:
            best_mse = error
            torch.save(model.state_dict(), f'./feature_extractor/models/{type(model).__name__}_{latent_dim}.pth')
            
        print(f'{epoch:>5} {total_loss / (epoch + 1):>25} {error:>15}')
        
    return best_mse, total_loss / (epoch + 1)
        
def evaluate_feature_extractor(model : nn.Module, val_dataset, batch_size : int = 32):
    """
    Evaluate the feature extractor model on the given dataset.
    
    Args:
    - model: The feature extractor model to evaluate.
    - dataset: The dataset to evaluate on.
    - batch_size: Size of each evaluation batch.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            
            # Forward pass
            _, reconstructed = model(x)
            
            loss = nn.MSELoss()(reconstructed, x)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)
