import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn(1,6,512)
B, T, D = x.shape
h = 4
head_dim = D // h

wq, wk, wv, wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)

q, k, v = wq(x), wk(x), wv(x)

q = q.reshape(B, h, T, head_dim).transpose(1, 2)
k = k.reshape(B, h, T, head_dim).transpose(1, 2)
v = v.reshape(B, h, T, head_dim).transpose(1, 2)

attention = (F.softmax(((q @ k.transpose(-2, -1)) / (D ** 0.5)), dim=-1)) @ v

mha = wo(attention.transpose(1, 2).reshape(B, T, D))

print(mha.shape)
