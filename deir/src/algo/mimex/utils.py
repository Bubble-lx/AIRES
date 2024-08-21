
import os
import torch
import torch.nn as nn
from src.algo.mimex.mvp.backbones import vit
import matplotlib.pyplot as plt
import random
import re
import time
import os
import numpy as np

_HOI_MODELS = {
    "maevit-s16": "mae_pretrain_hoi_vit_small.pth",
}
_IN_MODELS = {
    "vit-s16": "sup_pretrain_imagenet_vit_small.pth",
    "maevit-s16": "mae_pretrain_imagenet_vit_small.pth",
}
pretrain_dir = "/deir-main/src/algo/intrinsic_rewards/pretrain"
pretrain_fname = "mae_pretrain_hoi_vit_small.pth"
pretrain_type = 'hoi'
model_type = 'maevit-s16'
freeze = True
emb_dim = 128 
# ----------------------------------------------------------------
input_dim = 64
decoder_embed_dim = 8
mask_ratio = 0.7
# ----------------------------------------------------------------
# MIMEx setings
input_dim_mimex = 16
decoder_embed_dim_mimex = 8
anneal_weight = 1.0
image_counter = 0
save_path = "/deir-main/src/algo/mimex/heatmap"
#  ---------------------------------------------------------------

# Draw an attn heatmap
def draw_heat_map(num, attn):

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    n = 90
    m = 8192 // n
    truncated_length = n * m
    vmin_value = 0
    vmax_value = 0.7
    for j in range(4):
        for i in range(4):
            data = attn[j][:, i, 1, 2].detach().cpu().numpy()
            data_draw = data[:truncated_length].reshape(n, m)
            plt.figure(figsize=(8, 6))
            plt.imshow(data_draw, cmap='viridis', interpolation='nearest', vmin=vmin_value, vmax=vmax_value)
            plt.colorbar()
            save_file_path = os.path.join(save_path, f"depth{j}_heard{i}_{num}.png")
            plt.savefig(save_file_path)
            plt.close()
    print("showing heatmap")

    