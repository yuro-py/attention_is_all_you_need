import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn(1, 6, 512)

B, T, D = x.shape
h = 4
head_dim = D // h

wq, wk, wv, wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)

q, k, v = wq(x), wk(x), wv(x)

q = q.reshape(B, T, h, head_dim).transpose(1, 2)
k = k.reshape(B, T, h, head_dim).transpose(1, 2)
v = v.reshape(B, T, h, head_dim).transpose(1, 2)

# Attention scores
scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)

# Causal mask (upper triangle)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))

# Masked attention
attention = F.softmax(scores, dim=-1) @ v

mha = wo(attention.transpose(1, 2).reshape(B, T, D))

print(mha.shape)
