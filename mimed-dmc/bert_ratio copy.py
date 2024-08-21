from functools import partial

import torch
import torch.nn as nn
import math

from timm.models.vision_transformer import Block

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
        mask_ratio=0.7, norm_loss=False, use_cls=False):
        super().__init__()

        # self.mask_ratio = mask_ratio
        self.norm_loss = norm_loss # 是否在损失计算中使用标准化
        self.use_cls = use_cls # 是否在序列前加入CLS（分类）标记，通常用于分类任务。

        # --------------------------------------------------------------------------
        # BERT encoder specifics

        self.encoder_embed = nn.Linear(feature_dim, embed_dim) # feature_dim 为39200， embed_dim位为128

        self.action_embed = nn.Linear(1, 128)

        self.next_obs_embed = nn.Linear(39200, 128)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # sum of features and positional embeddings
        self.pos_embed = PositionalEncoding(
            d_model=embed_dim, max_len=seq_len)

        self.pos_embed_after = PositionalEncoding(
            d_model=embed_dim, max_len=15)

        # mlp_ratio：Transformer块中前馈网络的隐藏层大小与嵌入维度的比率。
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # BERT decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # 掩码标记，用于在解码过程中替换被掩码的元素。

        self.decoder_pos_embed = PositionalEncoding(
            d_model=decoder_embed_dim, max_len=seq_len)
        
        self.decoder_pos_embed_after = PositionalEncoding(
            d_model=decoder_embed_dim, max_len=15)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, feature_dim, bias=True) # 将解码器的输出映射回原始特征空间的线性层。
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

    def forward_encoder(self, x, seq_act, seq_next_obs):
        # (B, L, feature_dim) -> (B, L, embed_dim)
        x = self.encoder_embed(x) # 输入为[1024, 5, 39200]输出为[1024, 5, 128]

        a = self.action_embed(seq_act) # 输出为[1024, 5, 128]

        seq_next_obs = self.next_obs_embed(seq_next_obs)

        # 声明分割符号
        separator_token_embedding = torch.zeros(128, device="cuda")
        # # 调整形状以匹配输入序列的维度，并重复以匹配批次大小和序列长度
        separator_token_embedding = separator_token_embedding.repeat(1024, 5, 1)
        # 按照obs、分隔符、action的顺序拼接
        # x = torch.cat((x, separator_token_embedding, a), dim=2) # 输出[1024, 15, 128]
        # 初始化一个空列表来保存每次循环中拼接的结果
        concatenated_slices = []

        # 循环遍历第二维度的每一个索引
        for i in range(5):  # 因为第二维度的大小是5
            # 从每个Tensor中取出第i个切片，并立即进行拼接
            slice_a = x[:, i:i+1, :]
            slice_b = separator_token_embedding[:, i:i+1, :]
            slice_c = a[:, i:i+1, :]
            # 将三个切片沿第二维度拼接
            concatenated_slice = torch.cat([slice_a, slice_b, slice_c], dim=1)
            # 将拼接好的切片添加到列表中
            concatenated_slices.append(concatenated_slice)
        # 最后，将所有拼接好的切片沿第二维度再次拼接，形成最终的Tensor
        intput_cat = torch.cat(concatenated_slices, dim=1)
        # x = result # 输出[1024, 15, 128]

        # add pos embed
        intput_cat = intput_cat + self.pos_embed_after(intput_cat) # 输出[1024, 15, 128]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x)

        if self.use_cls:
            # append cls token
            cls_token = self.cls_token  # pos emb can be ignored since it's 0
            cls_tokens = cls_token.expand(intput_cat.shape[0], -1, -1) # cls_tokens是[1024, 1, 128]
            intput_cat = torch.cat((cls_tokens, intput_cat), dim=1) # x是[1024, 16, 128]

        # apply Transformer blocks
        for blk in self.blocks:
            intput_cat = blk(intput_cat)
        intput_cat = self.norm(intput_cat) # x是[1024, 16, 128]

        # return x, mask, ids_restore
        return intput_cat, seq_next_obs

    def forward_decoder(self, x):
        """
        Decoder processing adjusted for direct input without masking.
        """
        x = self.decoder_embed(x) # 输出x是[1024, 16, 64]

        if self.use_cls:
            # add pos embed to non-cls-tokens
            x[:, 1:, :] = x[:, 1:, :] + self.decoder_pos_embed_after(x[:, 1:, :]) # 输出x是[1024, 16, 64]
        else:
            # add pos embed to all tokens
            x = x + self.decoder_pos_embed_after(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x) # 输出x是[1024, 16, 39200]

        if self.use_cls:
            # remove cls token
            x = x[:, 1:, :]  # 输出x是[1024, 15, 39200]

        return x

    def forward(self, x, seq_act, seq_next_obs, keep_batch=False):
        """
        Model forward pass simplified to exclude random masking logic.
        """
        z, seq_next_obs_after = self.forward_encoder(x, seq_act, seq_next_obs)
        pred = self.forward_decoder(z)
        loss = self.forward_loss(seq_next_obs_after, pred, keep_batch)

        # Your loss calculation logic might need adjustment based on the task.
        # For simplicity, it's not included here.

        return loss

    def forward_loss(self, x, pred, keep_batch=True):
        """
        计算损失函数，不使用标准化，不使用掩码机制，保持批次维度。

        参数:
        x: [B, L, D] 真实值
        pred: [B, L, D] 预测值
        keep_batch: 保持批次维度，这里默认为True

        返回:
        损失值，保持批次维度
        """
        if self.norm_loss:
            # normalize loss
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6)**.5

        # 计算预测值和真实值之间的均方误差
        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # [B, L], mean loss per timestep

        # 对每个样本计算平均损失
        if keep_batch:
            loss = loss.mean(dim=-1)  # [B], mean loss per batch
        else:
            loss = loss.mean()  # mean loss across all batches and timesteps
        # 由于keep_batch=True，我们直接返回每个样本的损失，不进行进一步的平均化
        return loss

