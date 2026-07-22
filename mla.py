import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn(10,60,51200)
B, T, D = x.shape
h = 4
head_dim = D // h

# DeepSeek MLA: Add latent compression
latent_dim = D // 4  # Compress to 1/4 size

# Query/Key compression and up-projection
wq_compress = nn.Linear(D, latent_dim)
wq_up = nn.Linear(latent_dim, D)
wk_compress = nn.Linear(D, latent_dim)
wk_up = nn.Linear(latent_dim, D)
wv, wo = nn.Linear(D, D), nn.Linear(D, D)

# Compressed Q, K paths
q_latent = wq_compress(x)
q = wq_up(q_latent)
k_latent = wk_compress(x)
k = wk_up(k_latent)
v = wv(x)

# Standard MHA operations (but with latent-projected Q,K)
q = q.reshape(B, h, T, head_dim).transpose(1, 2)
k = k.reshape(B, h, T, head_dim).transpose(1, 2)
v = v.reshape(B, h, T, head_dim).transpose(1, 2)

attention = (F.softmax(((q @ k.transpose(-2, -1)) / (D ** 0.5)), dim=-1)) @ v

mha = wo(attention.transpose(1, 2).reshape(B, T, D))

print(mha.shape)
