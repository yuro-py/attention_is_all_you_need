"""import torch
import torch.nn.functional as F

x = torch.randn([1, 4, 1000])

B, T, C = x.shape

W_q = torch.randn(C, C)
W_k = torch.randn(C, C)
W_v = torch.randn(C, C)

Q = x @ W_q
K = x @ W_k
V = x @ W_v

unscaled_scores = Q @ K.transpose(-2, -1)
scaled_scores = unscaled_scores / C ** 0.5

weights = F.softmax(scaled_scores, dim = -1)

out =  weights @ V

print(out.shape)
"""

import time

start = time.perf_counter()

import torch
import torch.nn.functional as F

x = torch.randn(5,50,25000)

B, T, C = x.shape

W_q = torch.randn(C, C)
W_k = torch.randn(C, C)
W_v = torch.randn(C, C)

Q = x @ W_q
K = x @ W_k
V = x @ W_v

scores = Q @ K.transpose(-2, -1) # UNSCALED DOT-PRODUCT ATTENTION

scores = scores / (C ** 0.5) # SCALED DOT-PRODUCT ATTENTION

weights = F.softmax(scores, dim=-1)

out = weights @ V

print(out.shape)

time.sleep(2)

end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} seconds")
