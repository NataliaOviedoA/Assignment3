import load_dataset as data
import torch 
from torch import nn
import numpy as np
import gc
# Skeleton base 

# class MyRNN(nn.Module):
#     # Need the input_size, hidden_size and output_size
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MyRNN, self).__init__()
#         self.hidden_size = hidden_size
#         # For the 2 linear layers
#         #self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
#         #self.in2output = nn.Linear(input_size + hidden_size, output_size)
    
#     def forward(self, x, hidden_state):
#         combined = torch.cat((x, hidden_state), 1)
#         hidden = torch.sigmoid(self.in2hidden(combined))
#         output = self.in2output(combined)
#         return output, hidden
    
#     def init_hidden(self):
#         return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

# hidden_size = 256
# learning_rate = 0.001
# input_size = 0
# output_size = 0

# model = MyRNN(input_size, hidden_size, output_size)
# # Using the cross entropy loss
# criterion = nn.CrossEntropyLoss()
# # Which optimizer are we going to use?
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# num_epochs = 2
# print_interval = 3000

# training (forward, loss and backward)
'''
for epoch in range(num_epochs):
    random.shuffle(train_dataset)
    for i, (name, label) in enumerate(train_dataset):
        hidden_state = model.init_hidden()
        for char in name:
            output, hidden_state = model(char, hidden_state)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        if (i + 1) % print_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(train_dataset)}], "
                f"Loss: {loss.item():.4f}"
            )
'''













#If final is true, the function returns the canonical test/train split with 25 000 reviews in each.
#If final is false, a validation split is returned with 20 000 training instances and 5 000
#validation instances.

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data.load_imdb(final=False)

# The return values are as follows:
# x_train A python list of lists of integers. Each integer represents a word. Sorted from short to long.
# y_train The corresponding class labels: 0 for positive, 1 for negative.
# x_val Test/validation data. Laid out the same as x_train.
# y_val Test/validation labels
# i2w A list of strings mapping the integers in the sequences to their original words. i2w[141] returns the string containing word 141.
# w2i A dictionary mapping the words to their indices. w2i['film'] returns the index for the word "film".

# To have a look at your data (always a good idea), you can convert a sequence from indices to words as follows
#print([i2w[w] for w in x_train[141]])

# To train, you'll need to loop over x_train and y_train and slice out batches. 
# Each batch will need to be padded to a fixed length and then converted to a torch tensor. 
# Implement this padding and conversion to a tensor
#Batch size
batch_size = 250

#Counters
k = 0
j = batch_size
#Splitting the xtrain into batches.
matrix = []  
for i in range(80):
    matrix.append(x_train[k:j])
    k = j
    j = j + batch_size          

maxLength = 0
#Looping the matrix.
for i in matrix:
    #Getting the highest value from each value.
    current_length = max(len(x) for x in i)
    if current_length > maxLength:
        maxLength = current_length

#adding padding to each element of the list through a nested loop.
for i in matrix:
    for j in i:
        while len(j) < maxLength:
            j.append(0)

#Embeddong process good job natalia
embedding = nn.Embedding(99430, 300)
b1 = torch.tensor(matrix[0], dtype = torch.long)
b = embedding(b1)


#deleting variables from memory
del b1
del maxLength
gc.collect()


#First linear layer
linear1 = nn.Linear(300,300)
output1 = linear1(b)

#ReLU
relu = nn.ReLU()
output2 = relu(output1)

#Applying torchmax to reduce dimensionality
output3, _  = torch.max(output2, 1)


#Project down number of classes?
linear3 = nn.Linear(300,2)
output4 = linear3(output3)
print(np.shape(output4))


    
    
    
    
    
            