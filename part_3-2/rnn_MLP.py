import load_dataset as data
import torch 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


#DATA
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data.load_imdb(final=False)
(x_train2, y_train2), (x_test, y_test), (i2w2, w2i2), numcls = data.load_imdb(final=True)

#Class for the neural network
class Net(nn.Module):
    def __init__(self): 
        super().__init__()
        self.embedding = nn.Embedding(100000, 300)
        self.fc1 = nn.Linear(300,300)
        self.fc2 = nn.Linear(300,2)
    def forward(self,x): #algorithm for the forward propagation
        x = self.embedding(x)
        x = self.fc1(x)
        x = F.relu(x)
        x,_ = torch.max(x,1)
        x = self.fc2(x)
        return x
    
def accuracy(x,y):
    preds = torch.argmax(x,1)
    return (torch.sum(preds == y)/len(y)).item()


#Setting hyperparameters
num_epochs = 3
learning_rate = 0.001
batch_size = 50
batch_iter = 300
total = batch_size*batch_iter
Vbatch_size = 50
Vbatch_iter = 100
Vtotal = Vbatch_size * Vbatch_iter
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

#splitting y__test labels into batches
testlabels = []
k = 0
j = Tbatch_size
for i in range(Tbatch_iter):
    testlabels.append(y_test[k:j])  
    k = j
    j = j+Tbatch_size 

#splitting y_val labels into batches
vallabels = []
k = 0
j = Vbatch_size
for i in range(Vbatch_iter):
    vallabels.append(y_val[k:j])  
    k = j
    j = j+Vbatch_size 

#Padding xtrain
xtrain = x_train[0:total]
max_length =len(xtrain[-1])
for i in xtrain:
    while len(i)<max_length:
        i.append(0)

#Padding xvalue
vtrain = x_val[0:Vtotal]
max_length =len(vtrain[-1])
for i in vtrain:
    while len(i)<max_length:
        i.append(0)

#Padding xtest
ttrain = x_test[0:Ttotal]
max_length =len(ttrain[-1])
for i in ttrain:
    while len(i)<max_length:
        i.append(0)
        

#Data Inputs for training
xtrain = torch.tensor(xtrain, dtype = torch.long)
trainset = torch.utils.data.DataLoader(xtrain, batch_size)
labels = torch.tensor(labels, dtype = torch.long)

#Data Inputs for validation
x_valtrain = torch.tensor(vtrain, dtype = torch.long)
valset = torch.utils.data.DataLoader(x_valtrain,Vbatch_size)
vallabels = torch.tensor(vallabels, dtype = torch.long)

#Data Inputs for test
x_testtrain = torch.tensor(ttrain, dtype = torch.long)
testset = torch.utils.data.DataLoader(x_testtrain,Tbatch_size)
testlabels = torch.tensor(testlabels, dtype = torch.long)

#Initialize network, loss and optimizer
NeuralNet = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(NeuralNet.parameters(), lr=0.001)

#Epochs and baches
vector = [0]
valid_vector = [0]
test_vector = [0]
for epoch in range(num_epochs):
    count = 0
    acc = 0
    loss_mt = 0
    for i in trainset:
        NeuralNet.zero_grad() #set gradients to zero
        myNet = NeuralNet(i)#run forward
        loss = criterion(myNet, labels[count])#calculating loss
        print(accuracy(myNet,labels[count]))
        acc += accuracy(myNet,labels[count])
        count += 1
        loss.backward() #backward process
        optimizer.step() #iptimizer
    final_acc = acc/batch_iter
    # vector.append(final_acc)
    vector.append(final_acc)

    NeuralNet.eval()
    count = 0
    loss_mv = 0
    acc2 = 0
    for i in valset:
        myNet = NeuralNet(i) 
        loss = criterion(myNet, vallabels[count])
        #loss_mv +=loss.item()
        acc2 += accuracy(myNet,vallabels[count])
        print("Acc = {0}".format(accuracy(myNet,vallabels[count])))
        count += 1
        
    valid_vector.append(acc2/Vbatch_iter)

#Testing
count = 0
loss_mv = 0
acc2 = 0
for i in testset:
    myNet = NeuralNet(i) 
    loss = criterion(myNet, testlabels[count])
    acc2 += accuracy(myNet,testlabels[count])
    print("Acc = {0}".format(accuracy(myNet,testlabels[count])))
    count += 1
    
test_vector.append(acc2/Tbatch_iter)
print(test_vector)




plt.plot(vector, label = "Training")
plt.plot(valid_vector, color = 'r', label = "Validation")
plt.title("Accuracy Training/Validation")
plt.ylabel('Accuracy')
plt.xlabel("Epochs")
plt.legend()
plt.show()
    
    
    




    
    
    
    
    
            