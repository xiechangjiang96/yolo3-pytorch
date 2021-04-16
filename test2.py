import torch.nn as nn
import torch

m = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0, bias=False)
rand_data = torch.rand([1, 1, 4, 4])
m(rand_data)