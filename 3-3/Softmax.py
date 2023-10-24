import torch
from torch import nn
import matplotlib.pylab as plt

m = nn.Softmax(dim=1) # 各行でソフトマックス関数
x = torch.tensor([[1.0, 2.0, 3.0],
                    [3.0, 2.0, 1.0]])

y = m(x)

print(y)
