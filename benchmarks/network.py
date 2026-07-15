import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  '''
  Positional encoding using cosine and sine of the input.

  params:
    - L - number of frequencies
  '''
  def __init__(self, L) -> None:
    super().__init__()
    self.L = L
    self.powers = torch.tensor(2).pow(torch.arange(self.L))

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    '''
    Encode a tensor.

    params:
      - X - features to encode of shape (... x d)
    
    returns:
      - encoded features of shape (... x d * L * 2)
    '''
    dims = X.dim()
    device = X.device
    scaled = self.powers[(None,)*dims].to(device) * X.unsqueeze(-1)
    enc = torch.stack([torch.cos(scaled), torch.sin(scaled)], dim=-1).flatten(start_dim=dims-1)
    return enc
  
  import torch

class ResBlock(nn.Module):
  def __init__(self, size: int, activation=nn.Softplus, **kwargs) -> None:
    super().__init__()
    self.l0 = nn.Linear(size, size)
    self.l1 = nn.Linear(size, size)
    self.act = activation(**kwargs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    vals = self.act(self.l0(x))
    vals = self.act(self.l1(vals))
    return x + vals

class MLP(nn.Module):
  def __init__(self, sizes=[3, 64, 64, 1], activation=nn.Softplus, batchNorm=False, **kwargs) -> None:
    '''
    A simple MLP with given layer sizes.

    in: 
      - sizes - number of features in each layer (including in and out sizes)
      - activation - activation module to use
      - batchNorm - use batch normalization between layers and activations
      - **kwargs - additional keyword arguments to pass to the activation module
    '''
    super().__init__()
    # construct network from hidden_sizes
    layers = []
    for i in range(len(sizes)-1):
      layer = nn.Linear(sizes[i], sizes[i+1])
      layers.append(layer)
      if i < len(sizes) - 2:
        if batchNorm:
          layers.append(nn.BatchNorm1d(sizes[i+1]))
        layers.append(activation(**kwargs))
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

class SdfNet(torch.nn.Module):
  def __init__(self, d: int, L=4, hidden_sizes=[128, 128], beta=30) -> None:
    L = 4
    super().__init__()
    self.net = MLP(sizes=[d*L*2] + hidden_sizes + [1], activation=torch.nn.Softplus, beta=beta)
    self.encoding = PositionalEncoding(L)
    
  def forward(self, x) -> torch.Tensor:
    feats = self.encoding(x)
    vals = self.net(feats)
    return vals

class DeformNet(torch.nn.Module):
  def __init__(self, d: int, L=4, hidden_sizes=[128, 128], beta=30) -> None:
    L = 4
    super().__init__()
    self.net = MLP(sizes=[d*L*2] + hidden_sizes + [d], activation=torch.nn.Softplus, beta=beta)
    self.encoding = PositionalEncoding(L)
    self.net.net[-1].bias.data.zero_()
    self.net.net[-1].weight.data.zero_()
    
  def forward(self, x):
    feats = self.encoding(x)
    delta = self.net(feats)
    return x + delta
