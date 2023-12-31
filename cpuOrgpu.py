#%%time

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

cifar10_train = CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
cifar10_test = CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())

batch_size = 64
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=len(cifar10_test), shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
if torch.cuda.is_available():
    net.cuda()

loss_fnc = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

record_loss_train = []
record_loss_test = []

#x_test, t_test = iter(test_loader).next()
x_test, t_test = next(iter(test_loader))
if torch.cuda.is_available():
    x_test, t_test = x_test.cuda(), t_test.cuda()

for i in range(10):
    net.train()
    loss_train = 0
    for j, (x, t) in enumerate(train_loader):
        if torch.cuda.is_available():
            x, t = x.cuda(), t.cuda()
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    net.eval()
    y_test = net(x_test)
    loss_test = loss_fnc(y_test, t_test).item()
    record_loss_test.append(loss_test)
