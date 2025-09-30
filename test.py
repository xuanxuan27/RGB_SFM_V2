
import torch


x = torch.zeros(1, 3, 1, 5)   # shape: [1, 3, 1, 5]
print(x)
print(x.shape)                # torch.Size([1, 3, 1, 5])

y = torch.squeeze(x)          # 移除所有 size=1 -> shape [3, 5]
print(y)
print(y.shape)                # torch.Size([3, 5])

z = torch.squeeze(x, 0)       # 只移除第0維 -> shape [3, 1, 5]
print(z)
print(z.shape)                # torch.Size([3, 1, 5])

w = torch.squeeze(x, 1)       # 第1維大小是3，不是1 -> 不變
print(w)
print(w.shape)                # torch.Size([1, 3, 1, 5])