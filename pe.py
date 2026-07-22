import torch

T = 6      # number of tokens (sequence length)
C = 512    # embedding dimension

# token positions: [[0], [1], [2], [3], [4], [5]]
pos = torch.arange(T).unsqueeze(1)

# even feature indices: [0, 2, 4, ..., 510]
i = torch.arange(0, C, 2)

# compute the angles used by both sin and cos
angle = pos / (10000 ** (i / C))

# positional encoding matrix
pe = torch.zeros(T, C)

# even dimensions -> sin
pe[:, 0::2] = torch.sin(angle)

# odd dimensions -> cos
pe[:, 1::2] = torch.cos(angle)

print(pe.shape)      # torch.Size([6, 512])
print(pe)
