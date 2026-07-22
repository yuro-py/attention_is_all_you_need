import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn([1, 6, 512])

D = x.shape[-1]

W_q = nn.Linear(512, 512)
W_k = nn.Linear(512, 512)
W_v = nn.Linear(512, 512)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

attention_scores = (Q @ K.transpose(-2, -1)) / (D ** 0.5)

weights = F.softmax(attention_scores, dim = -1)

out = weights @ V

print(out.shape)
