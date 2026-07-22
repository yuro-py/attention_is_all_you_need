import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn([1, 6, 512])

B, T, D = x.shape

wq, wk, wv = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)

q, k, v = wq(x), wk(x), wv(x)

attention = (F.softmax((q @ k.transpose(-2, -1)) / (D ** 0.5), dim=-1)) @ v

print(attention.shape)
