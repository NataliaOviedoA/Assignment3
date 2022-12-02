import torch 
from torch import nn
import numpy as np
import load_dataset as data

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data.load_imdb(final=False)

batch_size = 1000

# Counters
k = 0
j = batch_size
# Splitting the xtrain into batches.
matrix = []  
for i in range(20):
    matrix.append(x_train[k:j])
    k = j
    j = j + batch_size

maxLength = 0
# Looping the matrix.
for i in matrix:
    # Getting the highest value from each value.
    current_length = max(len(x) for x in i)
    if current_length > maxLength:
        maxLength = current_length

# adding padding to each element of the list through a nested loop.
for i in matrix:
    for j in i:
        while len(j) < maxLength:
            j.append(0)

hidden_size = 300
learning_rate = 0.001
input_size = 20000
output_size = 20000

# Size of the dictionary of embedding
embedding_num = 99429
# Embedding dimension - the size of each embedding vector
embedding_dim = 300 
# Embedding module
embedding = nn.Embedding(embedding_num, embedding_dim)
# Here - should be a loop over the 20 batches of the matrix
# Only for matrix[0] as an example
batch = torch.tensor(matrix[0], dtype=torch.long)
# Batch with embedding 
batch_embedding = embedding(batch)

# Linear dimention for second layer after embedding
hidden_dim = nn.Linear(embedding_dim, hidden_size)

# Hidden layer
hidden = hidden_dim(batch_embedding)

# Relu activation for non-linearity
hidden = torch.relu(hidden)

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
out, hidden = model.forward(here2)