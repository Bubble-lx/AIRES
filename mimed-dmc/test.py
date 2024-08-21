import torch

# Assuming the shape of attn is [1024, 4, 11, 11], we simulate it using random data here
attn = torch.rand(1024, 4, 11, 11)

# Extract specific elements
selected_attn = attn[:, :, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]

# Take the average of each attention head
mean_attn = selected_attn.mean(dim=1)

mean_attn.shape, mean_attn
