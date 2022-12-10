import load_dataset as data
import torch
from torch import nn
import numpy
import torch.distributions as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

### The ndfa and brackets dataset ###
x_train, (i2w, w2i) = data.load_ndfa(n=150_000) #batches (150, 10000, 188), chars 15
#x_train, (i2w, w2i) = data.load_brackets(n=150_0000) # batches (150,10000,1024), char 6

### The model ###
class LSTM(nn.Module):
    def __init__(self, embedding_num, embedding_dim,vocab,hidden_size,batch_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab = vocab
        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(embedding_num, embedding_dim)
        self.layer_LSTM = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.hidden_dim = nn.Linear(hidden_size, vocab)
        self.init_weights()

    def forward(self,batch,h_layer):
        seq_embed = self.embedding(batch) # Shape goes from (batch ,time) to (batch ,time,32)
        seq_embed = seq_embed.float()
        output, (h,c) = self.layer_LSTM(seq_embed, h_layer)  # (batch ,time,16)
        hidden = self.hidden_dim(output) # (batch ,time,15)
        return hidden, (h,c)

    def init_weights(self):
        ### Initialize weights  ###
        # Bias
        self.hidden_dim.bias.data.fill_(0)
        # hidden_dim weights
        self.hidden_dim.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        ### Initializes hidden state ###
        # Weigths for 1 batch * batch size * hidden size
        weight = next(self.parameters()).data
        return (weight.new(1, n_seqs, self.hidden_size).zero_(), # hidden state and cell state of LSTM
                weight.new(1, n_seqs, self.hidden_size).zero_())


### Batching and padding ###
batch_size = 60
number_batches = 2500
k = 0
j = batch_size
# Splitting the xtrain into batches.
matrix = []
#for i in range(150):
for i in range(number_batches):
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
maxLength = maxLength+2 #because max length changes from 158 to 160 with the tokens
# adding padding to each element of the list through a nested loop.
for i in matrix:
    for j in i:
        #Adding the start and end index token to each list
        j.insert(0, w2i[ '.start'])
        j.append(w2i[ '.end'])
        while len(j) < maxLength:
            j.append(0)
### Batching and padding done###


def sample(lnprobs, temperature):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome logits
    :param temperature: Sampling temperature. 1.0 follows the given
    distribution, 0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()

epochs = 50
hidden_size =16
### To know which data we are working with ###
if len(i2w) == 15:
    vocab = 15
if len(i2w) == 6:
    vocab = 6
learning_rate=0.001
embedding_num = len(i2w)# Size of the dictionary of embedding which is 15
embedding_dim = 32 # Embedding dimension - the size of each embedding vector
model = LSTM(embedding_num, embedding_dim,vocab,hidden_size,batch_size) # The model
#Uses nn.LogSoftmax() and nn.NLLoss() in one single class; extra softmax is not needed
criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
score = 1
LL_train = []

for i in range(epochs):
    running_loss = 0.0
    h = model.init_hidden(batch_size)
    ### Suffling data ###
    matrix = numpy.array(matrix)
    indices = numpy.random.permutation(number_batches)
    matrix = matrix[indices]
    matrix = matrix.tolist()
    batch_norm = []
    matches, total = 0, 0
    total_LL = []
    # Different patterns that can be used for samplinG
    seq = [w2i['.start'], w2i['s']]
    seq = torch.tensor(seq, dtype=torch.long)
    count = 0
    for batch_num in range(len(matrix)):
        model.train()
        running_loss = 0.0
        h = tuple([each.data for each in h])
        model.zero_grad()

        ## Making tensors ###
        batch = torch.tensor(matrix[batch_num], dtype=torch.long)
        target = batch.clone().detach()
        target = torch.tensor(target, dtype=torch.float)
        zero = torch.tensor([0])

        ### target is the bacth, shifted one token to the left ###
        for row in range(len(target)):
            target[row] = torch.cat((target[row][0:1], target[row][2:],zero)) #removes first token after begin
        for row in range(len(batch)):
            for item_pos in range(len(batch[row])):
                if i2w[batch[row][item_pos]] == ".end":
                    batch[row] = torch.cat((batch[row][0:item_pos-1], batch[row][item_pos:],zero)) #removes last token before end
                    break
        ### forward and backward ###
        hidden,h = model.forward(batch,h)
        hidden = torch.reshape(hidden, (batch_size, vocab, maxLength))
        target = target.long()
        loss = criterion(hidden.view(maxLength* batch_size, vocab), target.view(-1))
        loss.backward()

        ### grad clipping ###
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        norm_grad = model.hidden_dim.weight.grad.norm()
        optimizer.step()
        ### Adding loss and gradient norm for visualization ###
        running_loss += loss.item()
        if batch_num == 0:
            LL_train += [running_loss / number_batches]
        if batch_num % 250 == 0:
            print(f'[{i + 1}, {batch_num + 1:5d}] loss: {running_loss / number_batches:.10f}')
        total_LL += [running_loss / number_batches]
        batch_norm += [norm_grad.numpy()]
    ### Calculating the average loss per epoch ###
    LL_train += [numpy.mean(total_LL)]
   ### Sampling and seed sequence ###
    model.eval()
    with torch.no_grad():
        h = model.init_hidden(1)
        while count != 10: # Each seed gets a max length of 10
            # Making a single batch of the sequence to feed in model
            seq = torch.tensor(seq , dtype=torch.long)
            seq = torch.unsqueeze(seq, dim=0)
            h = tuple([each.data for each in h])
            outputs,h = model.forward(seq,h)
            # Using the last output to sample the next token
            score = sample(outputs[0, -1, :], temperature=0.5)
            # So it can be added to the sequence
            score = torch.unsqueeze(score, dim=0)
            print("The character that is sampled", i2w[score])
            # Adding the sampled token to  the sequence
            seq = torch.cat((seq[0], score))
            count += 1
            # If the sampled token is and end token, the sequence has a max length
            if i2w[score] == ".end":
                break
        print("the sequence is")
        print(''.join([i2w[i] for i in seq]))

### Visualizing the loss and grad norm
x1 = numpy.linspace(0, epochs, len(LL_train))
plt.plot(x1, LL_train, label="Training loss")
plt.legend()
plt.show()
x4 = numpy.linspace(0, epochs, len(batch_norm))
plt.plot(x4, batch_norm, label="grad norm")
plt.legend()
plt.show()


