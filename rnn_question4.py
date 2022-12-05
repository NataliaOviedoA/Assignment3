import torch 
from torch import nn
import numpy as np
import torch.nn.functional as F
import load_dataset as data
import torch.optim as optim
from torch.autograd import Variable 
import matplotlib.pyplot as plt

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data.load_imdb(final=True)

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
num_epochs = 2
learning_rate = 0.001
batch_size = 50
batch_iter = 300
total = batch_size*batch_iter
Tbatch_size = 50
Tbatch_iter = 300
Ttotal = Tbatch_size * Tbatch_iter
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

#splitting test_val labels into batches
testlabels = []
k = 0
j = Tbatch_size
for i in range(Tbatch_iter):
    testlabels.append(y_val[k:j])  
    k = j
    j = j+Tbatch_size 

#Padding xtrain
xtrain = x_train[0:total]
max_length =len(xtrain[-1])
for i in xtrain:
    while len(i)<max_length:
        i.append(0)

#Padding xvalue
ttrain = x_val[0:Ttotal]
max_length =len(ttrain[-1])
for i in ttrain:
    while len(i)<max_length:
        i.append(0)

#Data Inputs for training
xtrain = torch.tensor(xtrain, dtype = torch.long)
trainset = torch.utils.data.DataLoader(xtrain, batch_size)
labels = torch.tensor(labels, dtype = torch.long)

#Data Inputs for validation
x_valtrain = torch.tensor(ttrain, dtype = torch.long)
validset = torch.utils.data.DataLoader(x_valtrain,Tbatch_size)
validlabels = torch.tensor(testlabels, dtype = torch.long)

#Initialize network, loss and optimizer
NeuralNet = Net(input_size, hidden_size, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(NeuralNet.parameters(), lr=0.001)


#Epochs and baches
vector = [0]
valid_vector = [0]
for epoch in range(num_epochs):
    count = 0
    acc = 0
    loss_mt = 0
    for i in trainset:
        NeuralNet.zero_grad() #set gradients to zero
        myNet = NeuralNet(i, None)#run forward
        loss = criterion(myNet, labels[count])#calculating loss
        print(accuracy(myNet,labels[count]))
        acc += accuracy(myNet,labels[count])
        # loss_mt += loss.item()
        # print(loss.item())
        count += 1
        loss.backward() #backward process
        optimizer.step() #iptimizer
    final_acc = acc/batch_iter
    # vector.append(final_acc)
    vector.append(final_acc)

count = 0
loss_mv = 0
acc2 = 0
for i in validset:
    myNet = NeuralNet(i, None) 
    loss = criterion(myNet, validlabels[count])
    #loss_mv +=loss.item()
    acc2 += accuracy(myNet,validlabels[count])
    print("Acc = {0}".format(accuracy(myNet,validlabels[count])))
    count += 1
    
valid_vector.append(acc2/Tbatch_iter)
print(valid_vector)

'''
plt.plot(vector, label = "Training")
plt.plot(valid_vector, color = 'r', label = "Validation")
plt.title("Accuracy Training/Validation")
plt.ylabel('Accuracy')
plt.xlabel("Epochs")
plt.legend()
plt.show()
'''    


