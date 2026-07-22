import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn(10, 60, 5120)

B, T, D = x.shape
num_heads = 8
head_dim = (D // num_heads)

wq, wk, wv, wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)

q, k, v = wq(x), wk(x), wv(x)

# Split into heads
q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)
v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)

# Self-attention for every head
attention = (F.softmax((q @ k.transpose(-2, -1)) / (head_dim**0.5), dim=-1)) @ v

# Merge heads
mha = wo(attention.transpose(1, 2).reshape(B, T, D))

print(mha.shape)

"""
explain this minimal implementation of classical MHA. also the math formula for MHA in the attention paper in an easy manner.

my explanation:-
import torch, functional and neural net.
declare random x values with 100 sentences, with 600 words each, each word having 5120 dimension of embedding layers.
declare B(batches), T(tokens), D(dimensions) and assign the shape of x and individual corresponding shape values to them.
num_heads = 8 heads into which the qkv will get divided into, to perform MHA. head_dim = the dimensions of each head((5120 / 8) in this case)
wq, wk, wv = weight projections for q, k and v formation. wo is the final linear transformation layer to get the final MHA answer(give a better explanation for it).
qkv are formed by passing the input x through the weighted linear layers.
now "# Split into heads", explain this properly, especially "reshape", of why and how, in easy manner.
then attention formula = softmax((q.k^t)/(d**0.5)) @ v
undo the reshape thing in "# Merge heads". explain this properly in an easy manner. here its transposing attentions 1 and 2 dims, then reshaping it back from 4 to 3 dims.
printing the shape of mha which shud match input shape."""
