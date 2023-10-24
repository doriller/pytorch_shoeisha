import torch
from torch import nn

y = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0]) # 出力
t = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]) # 正解

loss_func = nn.MSELoss() # 平均二乗誤差
loss = loss_func(y, t)
print(loss.item())
