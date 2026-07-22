import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn([1, 6, 512])
B, T, D = x.shape
h = 8
hd = D // h

def mha(q, k, v, wo, mask=None):
    q = q.reshape(B, T, h, hd).transpose(1, 2)
    k = k.reshape(B, T, h, hd).transpose(1, 2)
    v = v.reshape(B, T, h, hd).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    sa = F.softmax(scores, dim=-1) @ v
    return wo(sa.transpose(1, 2).reshape(B, T, D))

def ffn(x):
    net = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
    return net(x)

def encoder(x):
    wq, wk, wv, wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
    q, k, v = wq(x), wk(x), wv(x)

    norm1 = nn.LayerNorm(D)(x + mha(q, k, v, wo))
    norm2 = nn.LayerNorm(D)(norm1 + ffn(norm1))

    wk2, wv2 = nn.Linear(D, D), nn.Linear(D, D)
    return wk2(norm2), wv2(norm2)  # k, v handed to decoder's cross-attention

def decoder(y, enc_k, enc_v):
    wq, wk, wv, wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
    q, k, v = wq(y), wk(y), wv(y)

    mask = torch.triu(torch.ones([T, T]), diagonal=1).bool()
    norm1 = nn.LayerNorm(D)(y + mha(q, k, v, wo, mask))

    wq2 = nn.Linear(D, D)
    q2 = wq2(norm1)
    norm2 = nn.LayerNorm(D)(norm1 + mha(q2, enc_k, enc_v, wo))

    return nn.LayerNorm(D)(norm2 + ffn(norm2))

def transformer(x, y):
    enc_k, enc_v = encoder(x)
    d = decoder(y, enc_k, enc_v)

    linear = nn.Linear(D, D)
    probs = F.softmax(linear(d), dim=-1)
    return probs

print(transformer(x, x).shape)
