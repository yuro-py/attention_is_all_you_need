import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn([1, 6, 512])

B, T, D = x.shape

h = 8
hd = D // h

wq, wk, wv, wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
q, k, v = wq(x), wk(x), wv(x)

q = q.reshape(B, T, h, hd).transpose(1, 2)
k = k.reshape(B, T, h, hd).transpose(1, 2)
v = v.reshape(B, T, h, hd).transpose(1, 2)

def selfattn():
    sa = (F.softmax((q @ k.transpose(-2, -1)) / (D ** 0.5), dim=-1)) @ v
    return sa

def encoder():

    sa = selfattn()

    mha = wo(sa.transpose(1, 2).reshape(B, T, D))
    
    normalized = nn.LayerNorm(D)(mha)
    
    net = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            )

    ffnn = net(normalized)

    norm2 = nn.LayerNorm(D)(ffnn)

    wk, wv = nn.Linear(D, D), nn.Linear(D, D)
    k, v = wk(norm2), wv(norm2)

    return (k, v)











def decoder():

    scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)

    mask = torch.triu(torch.ones([T, T]),diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    
    sa = (F.softmax(scores, dim=-1)) @ v

    mha = wo(sa.transpose(1,2).reshape(B, T, D))

    norm = nn.LayerNorm(D)(mha)

    wq = nn.Linear(D, D)

    q = wq(norm)

    k, v = encoder() # IMP

    q = q.reshape(B, T, h, hd).transpose(1, 2)
    k = k.reshape(B, T, h, hd).transpose(1, 2)
    v = v.reshape(B, T, h, hd).transpose(1, 2)

    sa = selfattn()

    mha = wo(sa.transpose(1,2).reshape(B, T, D))

    norm = nn.LayerNorm(D)(mha)

    net = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            )

    ffnn = net(norm)

    norm = nn.LayerNorm(D)(ffnn)

    return norm


def transformer():
    net = nn.Linear(D, D)
    d = decoder()
    linear_output = net(d)
    probabilities = F.softmax(linear_output)
    
    return probabilities.shape

transformer()
