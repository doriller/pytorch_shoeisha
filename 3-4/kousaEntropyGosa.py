import torch
from torch import nn

# ソフトマックス関数への入力
x = torch.tensor([[1.0, 2.0, 3.0], # 入力1
                    [3.0, 2.0, 1.0]]) # 入力2

# 正解 (one-hot表現における1の位置)
t = torch.tensor([2, # 入力1に対応する正解
                    0]) # 入力2に対応する正解

loss_func = nn.CrossEntropyLoss() # ソフトマックス関数 + 交差エントロピー誤差
loss = loss_func(x, t)
print(loss.item())
