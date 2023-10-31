import torch
from torch import nn
#import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt

m = nn.Sigmoid()  # シグモイド関数

x = torch.linspace(-5, 5, 50)
y = m(x)

plt.plot(x, y)
# .show()では表示されないので .savefig()で画像を保存
#plt.show()
plt.savefig('sigmoid.jpg')

#matplotlib.plot(x, y)
#matplotlib.show()
