import torch
from torch import nn
import matplotlib.pylab as plt

m = nn.ReLU() # ReLU

x = torch.linspace(-5, 5, 50)
y = m(x)

plt.plot(x, y)
plt.savefig('relu.jpg')
