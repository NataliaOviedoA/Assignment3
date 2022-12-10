import load_dataset as data
import torch
from torch import nn
import copy
from torch.autograd import Variable
import numpy

x_train, (i2w, w2i) = data.load_ndfa(n=150_0000)


#x_train, (i2w, w2i) = data.load_brackets(n=150_0000)

# Have 150000 lists of integers indices.
# Firs couple of lists are [5, 5] and evetually big lists of numbers

# Counters
batch_size = 10000 #150 batches of size 10000
k = 0
j = batch_size
# Splitting the xtrain into batches.
matrix = []
for i in range(150):
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
maxLength = maxLength+2 #because max length changes from 186 to 188 with the tokens
# adding padding to each element of the list through a nested loop.
for i in matrix:
    for j in i:
        #Adding the start and end index token to each list
        j.insert(0, w2i[ '.start'])
        j.append(w2i[ '.end'])
        while len(j) < maxLength:
            j.append(0)




#if we want tensors as input, making batches with variable size
#max_cols = max([len(row) for batch in matrix for row in batch])
#max_rows = max([len(batch) for batch in matrix])
#matrix = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in matrix]
#matrix = [row + [0] * (max_cols - len(row)) for batch in matrix for row in batch]
#for i in matrix:
    #for j in i:
        #Adding the start and end index token to each list
        #j.insert(0, w2i[ '.start'])
        #j.append(w2i[ '.end'])
#padded = [row + [0] * (max_cols - len(row)) for batch in padded for row in batch]
#padded = padded.view(-1, max_rows, max_cols)
#shuffled = padded.shuffle(buffer_size=5)
#print(shuffled)
#print(matrix[0])
epochs = 2
hidden_size =16
vocab =15
learning_rate=0.05
# Size of the dictionary of embedding which is 15
embedding_num = len(i2w)
# Embedding dimension - the size of each embedding vector
embedding_dim = 32
# Embedding module
embedding = nn.Embedding(embedding_num, embedding_dim)
#In every epoch, the list of batches needs to be shuffled

#If True, then the input and output tensors are provided as (batch, seq, feature)
# Applies RNN to an input sequence
layer_LSTM = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=1,batch_first=True)
# Linear dimension
hidden_dim = nn.Linear(hidden_size, vocab)
#Uses nn.LogSoftmax() and nn.NLLoss() in one single class; extra softmax is not needed
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(layer_LSTM.parameters(), lr=learning_rate)
seq = [w2i['.start'], w2i['('], w2i['('], w2i[')']]
print(seq)
for i in range(epochs):
    running_loss = 0.0
    # shuffle the list of batches
    matrix = numpy.array(matrix)
    indices = numpy.random.permutation(150)
    matrix = matrix[indices]
    matrix = matrix.tolist()
    for batch_num in range(len(matrix)):
        #The target is a batch where the 1st position (pos 0 is the begin token) is removed.
        # A 0 is added to the end to have an equal
        target = [row[:] for row in matrix[batch_num]] #copying a batch as the target
        for row in target:
            del row[1]
            row.append(0)
        #print(target)


        target = torch.tensor(target)
        batch = torch.tensor(matrix[batch_num], dtype=torch.long)
        #Shape goes from (10000,186) to (10000,186,32)
        # Batch with embedding
        batch_embedding = embedding(batch)

        # Forward
        # represents a node in a computational graph
        h_0 = Variable(torch.zeros(1, batch_size, hidden_size)) # hidden state
        c_0 = Variable(torch.zeros(1, batch_size, hidden_size)) # internal state
        #lstm with input, hidden, and internal state
        output, (final_hidden_state, final_cell_state) = layer_LSTM(batch_embedding, (h_0, c_0))
        optimizer.zero_grad()
        # Shape becomes (1, 10000, 16) because h=16
        # Hidden layer
        #hidden = final_hidden_state.view(-1, hidden_size)  # reshaping the data to [batchsize, class]
        hidden = hidden_dim(output)


        #Gives batch, time, and num charachters [10000, 186, 15]
        # Target shape is (10000, 186) so batch size and time
        #hidden = torch.squeeze(hidden)
        hidden = torch.reshape(hidden,(10000,15,188))
        target = torch.squeeze(target) #Makes N, 10000 batches
        #print(hidden.shape) 10000,15,186
        #print(target.shape) #N,batch size; 10000,186
        #loss = nn.CrossEntropyLoss()
        #output = loss(hidden, target)
        #target.unsqueeze_(-1)
        #target = target.expand(10000, 186 ,1 )
        #print(target.shape)

        #print(output.shape)
        loss = criterion(hidden, target)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if batch_num % 10 == 0:
            print(f'[{i + 1}, {batch_num + 1:5d}] loss: {running_loss / 10:.3f}')
        running_loss = 0.0

