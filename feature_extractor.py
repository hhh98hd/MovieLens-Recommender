import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim : int, latent_dim : int, filter_num : int = 4, kernel_size : int = 5):
        super(FeatureExtractor, self).__init__()
       
        #1 Half-Convolutional Encoder
        self._encoder = nn.Sequential(
            nn.Conv1d(1, filter_num, kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=filter_num, stride=filter_num),
            nn.Flatten(),
        )
        
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
