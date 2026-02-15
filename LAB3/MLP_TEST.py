import torch
from MyNeuralNet import MyNeuralNet
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt

print(torch.backends.mps.is_available())

x_train = [[-2,-1], [-1,-1], [-1,-2],
 [2,-1], [1,-1], [1,-2],
 [2,1], [1,1], [1,2]]
y_train = [[1, 0, 0], [1, 0, 0], [1, 0, 0],
 [0, 1, 0], [0, 1, 0], [0, 1, 0],
 [0, 0, 1], [0, 0, 1], [0, 0, 1]]

X_train = torch.tensor(x_train).float()
Y_train = torch.tensor(y_train).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train = X_train.to(device)
Y_train = Y_train.to(device)

mynet = MyNeuralNet().to(device)
loss_func = nn.CrossEntropyLoss()
opt = SGD(mynet.parameters(), lr = 0.001) # “lr” is the learning rate.

n_epochs = 1000
loss_history = []
for _ in range(n_epochs):
    opt.zero_grad() # flush the previous epoch's gradients
    loss_value = loss_func(mynet(X_train),Y_train) # compute loss
    loss_value.backward() # perform back-propagation
    opt.step() # update the weights according to the gradients computed

    loss_history.append(loss_value.detach().cpu().numpy())


#for par in mynet.parameters():
    #print(par)

#Test data
x_test = [[-2,-2], [2,-2], [2,2]]
y_test = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # This means that the test labels are [0, 1, 2]
X_test, Y_test = torch.tensor(x_test).float().to(device), torch.tensor(y_test).float().to(device)
Y_pred = mynet(X_test)
print(Y_pred.cpu().detach().numpy())

print(torch.argmax(Y_pred, dim=1).cpu().numpy())


plt.plot(loss_history)
plt.title('Loss variation')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()