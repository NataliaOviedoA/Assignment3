import torch 
from torch import nn
import numpy as np

a = torch.randn(4,5,6)


output, _  = torch.max(a, 1)
print(output.shape)

