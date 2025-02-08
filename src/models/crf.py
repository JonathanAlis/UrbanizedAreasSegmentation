import torch
import torch.nn as nn
from crfseg import CRF

model = nn.Sequential(
    nn.Identity(),  # your NN
    CRF(n_spatial_dims=2)
)

batch_size, n_channels, spatial = 10, 5, (100, 100)
x = torch.zeros(batch_size, n_channels, *spatial)
log_proba = model(x)
print(log_proba.shape)