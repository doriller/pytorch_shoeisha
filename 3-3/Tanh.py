import torch
from torch import nn
import matplotlib.pylab as plt

m = nn.Tanh() # tanh

x = torch.linspace(-5, 5, 50)
y = m(x)

plt.plot(x, y)
plt.savefig('tanh.jpg')
