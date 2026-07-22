import time

start = time.perf_counter()

import torch
import torch.nn.functional as F

x = torch.randn(5,50, 25000)
B, T, C = x.shape

C_kv = 3

W_q = torch.randn(C, C)

W_down_kv = torch.randn(C, C_kv)

W_up_k = torch.randn(C_kv, C)
W_up_v = torch.randn(C_kv, C)

Q = x @ W_q

compressed_kv = x @ W_down_kv

K = compressed_kv @ W_up_k
V = compressed_kv @ W_up_v

scores = Q @ K.transpose(-2, -1)
scores = scores / (C ** 0.5)
weights = F.softmax(scores, dim=-1)
out = weights @ V

print(out.shape)

time.sleep(2)

end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} seconds")
