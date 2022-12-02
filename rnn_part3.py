import torch 
from torch import nn
import numpy as np


class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super().__init__()
        self.lin1 = nn.Linear(insize + hsize , hsize)
        self.lin2 = nn.Linear(insize + hsize, outsize)

    def forward(self, x, hidden=None):
        b, t, e = x.size()

        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)

        outs = []
        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            hidden = torch.sigmoid(self.lin1(inp))
            out = self.lin2(inp)
            outs.append(out[:, None, :])
        
        return torch.cat(outs, dim=1), hidden

model = Elman()

here = np.random.rand(1000,2154,300)
here2 = torch.tensor(here, dtype=torch.long)
print(here2.shape)
out, hidden = model.forward(here2)
print(out.shape)
print(hidden.shape)