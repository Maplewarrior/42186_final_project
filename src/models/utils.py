import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape  # Target shape

    def forward(self, x):
        return x.view((-1, ) + self.shape)  # (-1, ) infers the batch size
