import math
import torch
import torch.nn as nn
from functools import partial
from src.algo.mimex.mvp.ppo.vision_transformer import Block

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # compute the positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe.require_grad = False
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.pe[:, :x.size(1)]


class BERT_RATIO(nn.Module):
    """
    MAE-like BERT.
    """
    def __init__(self, seq_len, feature_dim, embed_dim=128, depth=4,
        num_heads=4, decoder_embed_dim=64, decoder_num_heads=2, decoder_depth=1,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), dropout=0.,
        mask_ratio=0.7, norm_loss=False, use_cls=False, action_shape=5, attn_speed_way=1):
        super().__init__()

        # self.mask_ratio = mask_ratio
        self.norm_loss = norm_loss # standardization used in loss calculation or not
        self.use_cls = use_cls # Whether to add CLS (Classification) tags before the sequence, usually used for classification tasks

        # --------------------------------------------------------------------------
        # BERT encoder specifics
        self.encoder_embed = nn.Linear(feature_dim, embed_dim)
        self.action_embed = nn.Linear(action_shape, embed_dim)

        # self.next_obs_embed = nn.Linear(39200, embed_dim)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # sum of features and positional embeddings
        self.pos_embed = PositionalEncoding(
            d_model=embed_dim, max_len=seq_len)

        self.pos_embed_after = PositionalEncoding(
            d_model=embed_dim, max_len=10)

        self.linear_merge = nn.Linear(decoder_embed_dim * 2, embed_dim)

        # mlp_ratio: The ratio of hidden layer size to embedding dimension in feedforward networks of Transformer
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, attention_way=attn_speed_way)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # BERT decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask marker, used to replace masked elements during the decoding process
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = PositionalEncoding(
            d_model=decoder_embed_dim, max_len=seq_len)

        self.decoder_pos_embed_after = PositionalEncoding(
            d_model=decoder_embed_dim, max_len=15)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, attention_way=attn_speed_way)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, feature_dim, bias=True) #  Map the output of the decoder back to the linear layer of the original feature space.
        # --------------------------------------------------------------------------
        self.norm_loss = norm_loss

        self.initialize_weights()

    def initialize_weights(self):
        # torch.nn.init.normal_(self.mask_token, std=.02)

        if self.use_cls:
            torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, seq_act):
        # (B, L, feature_dim) -> (B, L, embed_dim)
        x = self.encoder_embed(x)
        a = self.action_embed(seq_act)

        concatenated_slices = []

        for i in range(5):  
            slice_a = x[:, i:i+1, :]
            slice_c = a[:, i:i+1, :]
            concatenated_slice = torch.cat([slice_a, slice_c], dim=1)
            concatenated_slices.append(concatenated_slice)
        intput_cat = torch.cat(concatenated_slices, dim=1)
        # add pos embed
        intput_cat = intput_cat + self.pos_embed_after(intput_cat)

        if self.use_cls:
            # append cls token
            cls_token = self.cls_token  # pos emb can be ignored since it's 0
            cls_tokens = cls_token.expand(intput_cat.shape[0], -1, -1)
            intput_cat = torch.cat((cls_tokens, intput_cat), dim=1)

        # apply Transformer blocks
        mean_attn_all = []
        for blk in self.blocks:
            intput_cat, attn = blk(intput_cat)
            selected_attn = attn[:, :, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
            mean_attn = selected_attn.mean(dim=1)
            mean_attn_all.append(mean_attn.unsqueeze(0))
        mean_attn_all = torch.cat(mean_attn_all, dim=0)
        summed_attn = mean_attn_all.sum(dim=0)

        intput_cat = self.norm(intput_cat)

        # return x, mask, ids_restore
        return intput_cat, summed_attn

    def forward_decoder(self, x):
        """
        Decoder processing adjusted for direct input without masking.
        """
        x = self.decoder_embed(x)

        if self.use_cls:
            # add pos embed to non-cls-tokens
            x[:, 1:, :] = x[:, 1:, :] + self.decoder_pos_embed_after(x[:, 1:, :])
        else:
            # add pos embed to all tokens
            x = x + self.decoder_pos_embed_after(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, attn = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.use_cls:
            # remove cls token
            x = x[:, 1:, :]

        return x

    def forward(self, x, seq_act, seq_next_obs, keep_batch=False):
        """
        Model forward pass simplified to exclude random masking logic.
        """
        z, summed_attn = self.forward_encoder(x, seq_act)
        merged_list = [z[:, 0:1, :]]
        for i in range(1, z.size(1) - 1, 2):
            pair = torch.cat([z[:, i:i+1, :], z[:, i+1:i+2, :]], dim=2)
            merged = self.linear_merge(pair)
            merged_list.append(merged)

        merged_z = torch.cat(merged_list, dim=1)
        # ----------------------------------------------------------------
        pred = self.forward_decoder(merged_z)
        loss = self.forward_loss(seq_next_obs, pred, keep_batch)

        # Your loss calculation logic might need adjustment based on the task.
        # For simplicity, it's not included here.

        return loss, summed_attn

    def forward_loss(self, x, pred, keep_batch=True):
        """
        Calculate the loss function without standardization or masking mechanism, 
        while maintaining the batch dimension.

        parameter:
        x: [B, L, D] true value
        pred: [B, L, D] predict value
        keep_batch: Maintain batch dimension, default to True here

        return:
        Loss value, maintain batch dimension
        """
        if self.norm_loss:
            # normalize loss
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6)**.5

        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # [B, L], mean loss per timestep

        if keep_batch:
            loss = loss.mean(dim=-1)  # [B], mean loss per batch
        else:
            loss = loss.mean()
        return loss
