import torch 
from torch import nn
import numpy as np
import torch.nn.functional as F
import load_dataset as data
import torch.optim as optim
from torch.autograd import Variable 

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data.load_imdb(final=False)

#Class for the neural network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers): 
        super().__init__()
        self.embedding = nn.Embedding(100000, 300)
        self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=n_layers) #lstm
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc2 = nn.Linear(hidden_size,2)
        self.num_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
    def forward(self,x, hidden): #algorithm for the forward propagation
        x = self.embedding(x)
        x = self.fc1(x)
        x = F.relu(x)
        x, hidden = self.rnn(x, hidden)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        x,_ = torch.max(output,1)
        x = self.fc2(x)
        return x
    
def accuracy(x,y):
    preds = torch.argmax(x,1)
    return (torch.sum(preds == y)/len(y)).item()


#Setting hyperparameters
input_size = 1254
n_layers = 1
hidden_size = 300
num_epochs = 3
learning_rate = 0.001
batch_size = 30
batch_iter = 500
#Counters
k = 0
j = batch_size

#Splitting the labels into batches
labels = []
k = 0
j = batch_size
for i in range(batch_iter):
    labels.append(y_train[k:j])  
    k = j
    j = j+batch_size  


#Padding xtrain
xtrain = x_train[0:15000]
max_length =len(xtrain[-1])
for i in xtrain:
    while len(i)<max_length:
        i.append(0)

#Data Inputs
xtrain = torch.tensor(xtrain, dtype = torch.long)
trainset = torch.utils.data.DataLoader(xtrain, batch_size)
labels = torch.tensor(labels, dtype = torch.long)

#Initialize network, loss and optimizer
NeuralNet = Net(input_size, hidden_size, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(NeuralNet.parameters(), lr=0.001)


#Epochs and baches
vector = []
for epoch in range(num_epochs):
    count = 0
    
    for i in trainset:
        NeuralNet.zero_grad() #set gradients to zero
        myNet = NeuralNet(i, None)#run forward
        loss = criterion(myNet, labels[count])#calculating loss
        print(accuracy(myNet,labels[count]))   
        vector.append(accuracy(myNet,labels[count]))
        count += 1
        loss.backward() #backward process
        optimizer.step() #iptimizer  
    print('This is the epoch') 

