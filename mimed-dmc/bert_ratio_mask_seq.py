import math
import torch
import torch.nn as nn
from functools import partial
from vision_transformer import Block, Block_origin

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


class BERT_RATIO_MASK(nn.Module):
    """
    MAE-like BERT.
    """

    def __init__(self, seq_len, feature_dim, embed_dim=128, depth=4,
        num_heads=4, decoder_embed_dim=64, decoder_num_heads=2, decoder_depth=1,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), dropout=0.,
        mask_ratio=0.7, norm_loss=False, use_cls=False, action_shape=5,
        use_next_obs=0, attn_selet_way=0, ratio_s2aanda2s=0.5, attn_ratio_weight=1.0, attn_speed_way=1):
        super().__init__()
        self.attn_speed_way = attn_speed_way

        self.mask_ratio = mask_ratio
        self.norm_loss = norm_loss # standardization used in loss calculation or not
        self.use_cls = use_cls # Whether to add CLS (Classification) tags before the sequence, usually used for classification tasks

        # --------------------------------------------------------------------------
        # BERT encoder specifics
        self.use_next_obs = use_next_obs

        # print("embedding dim")

        self.encoder_embed = nn.Linear(feature_dim, embed_dim)

        self.action_embed = nn.Linear(action_shape, embed_dim)
        if self.use_next_obs:
            self.next_obs_embed = nn.Linear(feature_dim, embed_dim)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # sum of features and positional embeddings
        self.pos_embed = PositionalEncoding(
            d_model=embed_dim, max_len=seq_len)

        self.pos_embed_after = PositionalEncoding(
            d_model=embed_dim, max_len=2 * seq_len + 1)

        self.linear_merge = nn.Linear(embed_dim * 2, embed_dim)

        # mlp_ratio:The ratio of hidden layer size to embedding dimension in feedforward networks of Transformer
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, attn_speed_way=attn_speed_way, attn_selet_way=attn_selet_way, use_cls=use_cls)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # --------------------------------------------------------------------------
        # BERT decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = PositionalEncoding(
            d_model=decoder_embed_dim, max_len=seq_len)

        self.decoder_pos_embed_after = PositionalEncoding(
            d_model=decoder_embed_dim, max_len=seq_len)

        self.decoder_blocks = nn.ModuleList([
            Block_origin(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # Map the output of the decoder back to the linear layer of the original feature space.
        self.norm_loss = norm_loss

        self.seq_len = seq_len
        self.ratio_s2aanda2s = ratio_s2aanda2s # the ratio of s2a and a2s
        self.attn_selet_way = attn_selet_way # 0 is s2a, 1 is s2aANDa2s, 2 is a2s
        self.attn_ratio_weight = attn_ratio_weight

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

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch 16, length 5, dim 8
        len_keep = int(L * (1 - self.mask_ratio)) # len_keep is 1

        noise = torch.rand(N, L, device=x.device)  # noise [8192, 5]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove. ids_shuffle [8192, 5]
        ids_restore = torch.argsort(ids_shuffle, dim=1) # ids_restore [8192, 5]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # [8192, 1]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # [8192, 1, 8]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0 # mask [8192, 5]
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # [8192, 5])

        return x_masked, mask, ids_restore # x_masked [8192, 1, 8], mask [8192, 5], ids_restore [8192, 5]

    def forward_encoder(self, x, seq_act, seq_next_obs):
        # (B, L, feature_dim) -> (B, L, embed_dim)
        x = self.encoder_embed(x)
        a = self.action_embed(seq_act)

        if self.use_next_obs:
            last_next_obs = self.next_obs_embed(seq_next_obs[:,-1:,:])

        concatenated_slices = []
        for i in range(self.seq_len):
            slice_a = x[:, i:i+1, :]
            slice_c = a[:, i:i+1, :]
            concatenated_slice = torch.cat([slice_a, slice_c], dim=1)
            concatenated_slices.append(concatenated_slice)
        if self.use_next_obs:
            concatenated_slices.append(last_next_obs)
        intput_cat = torch.cat(concatenated_slices, dim=1)
        # add pos embed
        intput_cat = intput_cat + self.pos_embed_after(intput_cat)

        if self.use_cls:
            # append cls token
            cls_token = self.cls_token  # pos emb can be ignored since it's 0
            cls_tokens = cls_token.expand(intput_cat.shape[0], -1, -1)
            intput_cat = torch.cat((cls_tokens, intput_cat), dim=1)

        attn_all = []
        for blk in self.blocks:
            intput_cat, attn = blk(intput_cat) # intput_cat:[16384, 11, 64]
            attn_all.append(attn)

        attn_return = []
        attn_return = torch.stack(attn_all, dim=0) 
        attn_return = torch.mean(attn_return, dim=0) 
        attn_return = torch.mean(attn_return, dim=1) 

        intput_cat = self.norm(intput_cat)
        return intput_cat, attn_return, attn_all

    def forward_decoder(self, x, ids_restore):
        """
        Decoder processing adjusted for direct input without masking.
        """
        x = self.decoder_embed(x) # [16, 2, 8]

        if self.use_cls:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # mask_tokens[16, 4, 8]
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle #x_[8192, 5, 8]
            x = torch.cat([x[:, :1, :], x_], dim=1)
            x[:, 1:, :] = x[:, 1:, :] + self.decoder_pos_embed(x[:, 1:, :])
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = x + self.decoder_pos_embed(x)

        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        if self.use_cls:
            x = x[:, 1:, :]

        return x

    def forward(self, x, seq_act, seq_next_obs, keep_batch=False):
        """
        Model forward pass simplified to exclude random masking logic.
        """
        z, summed_attn, attn_all = self.forward_encoder(x, seq_act, seq_next_obs)

        x_no_cls = z[:, 1:, :]
        cls_token = z[:, :1, :]

        z_maks, mask, ids_restore = self.random_masking(x_no_cls) # z_maks [16, 1, 8], mask[16, 3], ids_restore[16, 3]

        latent = torch.cat((cls_token, z_maks), dim=1) # latent[16, 2, 8]

        pred = self.forward_decoder(latent, ids_restore)

        loss = self.forward_loss(x_no_cls, pred, mask, keep_batch=keep_batch)

        summed_attn = summed_attn * self.attn_ratio_weight

        return loss, summed_attn, attn_all

    def forward_loss(self, x, pred, mask, keep_batch=True):
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
        # mean loss on removed timesteps
        if keep_batch:
            loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # (N,)
        else:
            loss = (loss * mask).sum() / mask.sum()

        return loss


