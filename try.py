import torch 
from torch import nn
n, d = 20000, 300
embedding = nn.Embedding(n, d, max_norm=True)
#W = torch.randn((m, d), requires_grad=True)
value = [[2,3,4,5,6], [1,2,3,4,5]]
idx = torch.tensor(value, dtype=torch.long)
b = embedding(idx)
print(b)

