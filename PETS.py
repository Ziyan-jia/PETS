import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import models.pnn_2d as pnn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import src.dataloader
ensemble_mean = 0
ensemble_var = 0
ensemble_size = 4          # Ensemble size per core
epochs = 80000
learning_rate = 1e-3
training_samples = 1000
validation_samples = training_samples
test_samples = 1
batch_size = 400
measurements = epochs//200  # Measure every n steps

# Define network architecture
input_dim = 21
output_dim = 14


def softplus( x):
    """ Compute softplus """
    softplus = torch.log(1 + torch.exp(x))
    # Avoid infinities due to taking the exponent
    softplus = torch.where(softplus == float('inf'), x, softplus)
    return softplus

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.hidden = torch.nn.Linear(21, 100)  # hidden layer
        self.predict = torch.nn.Linear(100, 14)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.hidden = torch.nn.Linear(21, 100)   # hidden layer
        self.predict = torch.nn.Linear(100, 14)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.hidden = torch.nn.Linear(21, 100)  # hidden layer
        self.predict = torch.nn.Linear(100, 14)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

class Net4(torch.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.hidden = torch.nn.Linear(21, 100)  # hidden layer
        self.predict = torch.nn.Linear(100, 14)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

data = src.dataloader.SimpleTwoDimensional()
ensemble = [Net1(),Net2(),Net3(),Net4()]
data.gather_train_samples(training_samples)
data.gather_validation_samples(validation_samples)
data.gather_test_samples(test_samples)

criterion = nn.MSELoss()


floss = []
elist = []
gradlist = []
epoch = 0
for model in ensemble:
    for epoch in range(epochs):
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        minibatch = data.get_batch(batch_size)
        model_in = minibatch[:, :21]
        y = minibatch[:, 21:]
        x = torch.from_numpy(model_in).float()
        output = model(x)
        mean, var = torch.split(output, output_dim // 2, dim=1)
        var = softplus(var)
        y = torch.from_numpy(y).float()
        diff = torch.sub(y, mean)
        #loss = torch.mean(torch.div(diff ** 2, 2 * var))
        #loss += torch.mean(0.5 * torch.log(var))
        loss = criterion(mean, y)
        loss_value = loss.data.cpu().numpy()
        optimizer.zero_grad()
        loss.backward()

        total_norm = 0
        for k in model.parameters():
            param_norm = k.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        gradlist.append(total_norm)
        optimizer.step()

        epoch += 1
        if epoch % 1000 == 0:
            print('Epoch:{}, loss:{:.6f}'.format(epoch, loss_value))
            floss.append(loss_value)
            elist.append(epoch)
        if loss_value <= 1e-1:
            break

    input_data = torch.from_numpy(data.test_data[:,:21]).float()
    output = model(input_data)
    mean, var = torch.split(output, output_dim // 2, dim=1)
    print(mean)
    ensemble_mean += mean
    ensemble_var += var + mean ** 2

ensemble_mean = ensemble_mean / ensemble_size
loss_ensemble = criterion(ensemble_mean, data.test_data[:,21:])
print(ensemble_mean)
print(loss_ensemble)