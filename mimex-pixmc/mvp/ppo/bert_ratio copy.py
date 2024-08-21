from functools import partial

import torch
import torch.nn as nn
import math
# from utils import draw_heat_map
from mvp.ppo.vision_transformer import Block

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
        mask_ratio=0.7, norm_loss=False, use_cls=False, action_shape=5,
        sequencelevel=0, attn_selet_way=0, ratio_s2aanda2s=0.5, attn_ratio_weight=1.0):
        super().__init__()

        # self.mask_ratio = mask_ratio
        self.norm_loss = norm_loss # 是否在损失计算中使用标准化
        self.use_cls = use_cls # 是否在序列前加入CLS（分类）标记，通常用于分类任务。

        # --------------------------------------------------------------------------
        # BERT encoder specifics

        self.encoder_embed = nn.Linear(feature_dim, embed_dim) # feature_dim 为39200， embed_dim位为64

        self.action_embed = nn.Linear(action_shape, embed_dim)

        self.next_obs_embed = nn.Linear(feature_dim, embed_dim)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # sum of features and positional embeddings
        self.pos_embed = PositionalEncoding(
            d_model=embed_dim, max_len=seq_len)

        self.pos_embed_after = PositionalEncoding(
            d_model=embed_dim, max_len=2 * seq_len + 1)

        self.linear_merge = nn.Linear(embed_dim * 2, embed_dim)

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
            d_model=decoder_embed_dim, max_len=seq_len)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, feature_dim, bias=True) # 将解码器的输出映射回原始特征空间的线性层。
        # --------------------------------------------------------------------------
        self.norm_loss = norm_loss

        self.seq_len = seq_len
        self.ratio_s2aanda2s = ratio_s2aanda2s # the ratio of s2a and a2s
        self.attn_selet_way = attn_selet_way # 0 is s2a, 1 is s2aANDa2s, 2 is a2s
        self.sequencelevel = sequencelevel # whether use multi sequence(self.seq_len) or a single sequence
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

    def forward_encoder(self, x, seq_act, seq_next_obs): #突然想起如果将obs*action = obs_next，注意力权重又会如何操作？
        # (B, L, feature_dim) -> (B, L, embed_dim)
        x = self.encoder_embed(x) # 输入为[16384, 5, 384]输出为[16384, 5, 128]

        a = self.action_embed(seq_act) # 输出为[16384, 5, 128]

        last_next_obs = self.next_obs_embed(seq_next_obs[:,-1:,:])

        concatenated_slices = []
        # 循环遍历第二维度的每一个索引
        for i in range(self.seq_len):  # 因为第二维度的大小是5
            # 从每个Tensor中取出第i个切片，并立即进行拼接
            slice_a = x[:, i:i+1, :]
            slice_c = a[:, i:i+1, :]
            # 将三个切片沿第二维度拼接
            concatenated_slice = torch.cat([slice_a, slice_c], dim=1)
            # 将拼接好的切片添加到列表中
            concatenated_slices.append(concatenated_slice)
        # 最后，将所有拼接好的切片沿第二维度再次拼接，形成最终的Tensor
        concatenated_slices.append(last_next_obs)
        intput_cat = torch.cat(concatenated_slices, dim=1) # 输出[1024, 10, 128]
        # add pos embed
        intput_cat = intput_cat + self.pos_embed_after(intput_cat) # 输出[1024, 10, 128]

        if self.use_cls:
            # append cls token
            cls_token = self.cls_token  # pos emb can be ignored since it's 0
            cls_tokens = cls_token.expand(intput_cat.shape[0], -1, -1) # cls_tokens是[1024, 1, 128]
            intput_cat = torch.cat((cls_tokens, intput_cat), dim=1) # x是[1024, 11, 128]


        attn_all = []
        for blk in self.blocks:
            intput_cat, attn = blk(intput_cat) # intput_cat:[16384, 11, 64]
            # draw_heat_map(attn) # 绘制注意力热力图
            # attn形状是[1024, 4, 11, 11],1-5时刻，各obs对动作action的注意力权重为[:, :, 1, 2]、[:, :, 3, 4]、[:, :, 5, 6]、[:, :, 7, 8]、[:, :, 9, 10]
            # selected_attn = attn[:, :, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]  # 形状为[1024, 4, 5]
            # test = attn[:, :, 9, 10]
            # mean_attn = selected_attn.mean(dim=1)  # 在第二个维度(注意力头)上取平均
            # mean_attn_all.append(mean_attn.unsqueeze(0)) # 在每个注意力头上取平均值得到[1024, 5]
            attn_all.append(attn)

        # apply Transformer blocks
        # --------------------------------
        # 第一种注意力权重：s1对a1的注意力权重
        # 第二种注意力权重：s1对a1的注意力权重以及a1对s2的注意力权重的平均值或者加权比如3：7加权，或者5：5加权
        # 第三种注意力权重：a1对s2的注意力权重
        # 读取s
        # s2a = 0
        # s2aANDa2s = 1
        # ratio_s2aanda2s = 0.5
        # a2s = 2
        attn_return = []
        # sequencelevel = 1 # single:RND\ICM\NGU\NovelD\DEIR, multi:MIMEx
        if self.use_cls:
            if self.attn_selet_way == 0: #如果采用状态对动作的注意力机制
                if self.sequencelevel:
                    # 第一种多批量 返回[16384, 5]
                    # attn[:, :, 1, 2]、attn[:, :, 3, 4]、attn[:, :, 5, 6]、attn[:, :, 7, 8]、attn[:, :, 9, 10],四个头在每个时刻上取均值
                    for j in range(self.seq_len): # Transformer的序列时刻
                        attn_tmp = []
                        for i in range(len(attn_all)): # Transformer的encoder的深度
                            attn_tmp.append(attn_all[i][:,:,2*j+1,2*j+2])
                        # 新增第0个维度上拼接并取均值
                        attn_tmp = torch.stack(attn_tmp, dim=0) # 返回attn_tmp为[4, 16384, 4]
                        attn_tmp = torch.mean(attn_tmp, dim=0) # 返回attn_tmp为[16384, 4] # 4表示注意力头数
                        attn_tmp = torch.mean(attn_tmp, dim=1) # 返回attn_tmp为[16384]
                        attn_return.append(attn_tmp) #返回5个[16384]，也就是每个时刻一个[16384]
                    # 是否拼接形成[16384, 5]
                    attn_return = torch.stack(attn_return, dim=1)
                else:
                    # 第一种单批量(RND、ICM等) 返回[16384]
                    # attn[:, :, 9, 10]表示s5对a5的注意力权重参数,seqlen是5
                    for i in range(len(attn_all)):
                        attn_return.append(attn_all[i][:,:,2*self.seq_len-1,2*self.seq_len]) # 返回：4个[16384, 4]
                    attn_return = torch.stack(attn_return, dim=0)
                    attn_return = torch.mean(attn_return, dim=0) # 返回一个[16384, 4],也就是最后一个序列的[16384, 4]
                    attn_return = torch.mean(attn_return, dim=1) # 返回一个[16384],也就是最后一个序列的[16384]
            # s1对a1的注意力权重以及a1对s2的注意力权重的平均值或者加权比如3：7加权，或者5：5加权
            elif self.attn_selet_way == 1:
                if self.sequencelevel:
                    for j in range(self.seq_len):
                        attn_tmp = []
                        for i in range(len(attn_all)):
                            attn_tmp.append(attn_all[i][:,:,2*j+1,2*j+2] * self.ratio_s2aanda2s + attn_all[i][:,:,2*j+2,2*j+3] * (1 - self.ratio_s2aanda2s))
                        attn_tmp = torch.stack(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=1)
                        attn_return.append(attn_tmp) #返回5个[16384, 4]，也就是每个时刻一个[16384, 4]，4表示注意力头数
                    attn_return = torch.stack(attn_return, dim=1)
                else:
                    # 单批量(RND、ICM等)：attn[:, :, 9, 10]表示s5对a5的注意力权重参数+attn[:, :, 10, 11]表示a5对s6的注意力权重
                    for i in range(len(attn_all)):
                        attn_return.append(attn_all[i][:,:,2*self.seq_len-1, 2*self.seq_len] * self.ratio_s2aanda2s + \
                                        attn_all[i][:,:,2*self.seq_len, 2*self.seq_len+1] * (1 - self.ratio_s2aanda2s))
                    attn_return = torch.stack(attn_return, dim=0)
                    attn_return = torch.mean(attn_return, dim=0) # 返回一个[16384, 4],也就是最后一个序列的[16384, 4]
                    attn_return = torch.mean(attn_return, dim=1) # 返回一个[16384],也就是最后一个序列的[16384]
            else:
                if self.sequencelevel:
                    for j in range(self.seq_len):
                        attn_tmp = []
                        for i in range(len(attn_all)):
                            attn_tmp.append(attn_all[i][:,:,2*j+2,2*j+3])
                        attn_tmp = torch.stack(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=1)
                        attn_return.append(attn_tmp) #返回5个[16384, 4]，也就是每个时刻一个[16384, 4]，4表示注意力头数
                    attn_return = torch.stack(attn_return, dim=1)
                else:
                    for i in range(len(attn_all)):
                        attn_return.append(attn_all[i][:,:,2*self.seq_len,2*self.seq_len+1])
                    attn_return = torch.stack(attn_return, dim=0)
                    attn_return = torch.mean(attn_return, dim=0) # 返回一个[16384, 4],也就是最后一个序列的[16384, 4]
                    attn_return = torch.mean(attn_return, dim=1) # 返回一个[16384],也就是最后一个序列的[16384]
        else:
            if self.attn_selet_way == 0: #如果采用状态对动作的注意力机制
                if self.sequencelevel:
                    for j in range(self.seq_len): # Transformer的序列时刻
                        attn_tmp = []
                        for i in range(len(attn_all)): # Transformer的encoder的深度
                            attn_tmp.append(attn_all[i][:,:,2*j,2*j+1])
                        # 新增第0个维度上拼接并取均值
                        attn_tmp = torch.stack(attn_tmp, dim=0) # 返回attn_tmp为[4, 16384, 4]
                        attn_tmp = torch.mean(attn_tmp, dim=0) # 返回attn_tmp为[16384, 4] # 4表示注意力头数
                        attn_tmp = torch.mean(attn_tmp, dim=1) # 返回attn_tmp为[16384]
                        attn_return.append(attn_tmp) #返回5个[16384]，也就是每个时刻一个[16384]
                    # 是否拼接形成[16384, 5]
                    attn_return = torch.stack(attn_return, dim=1)
                else:
                    for i in range(len(attn_all)):
                        attn_return.append(attn_all[i][:,:,2*self.seq_len-2,2*self.seq_len-1]) # 返回：4个[16384, 4]
                    attn_return = torch.stack(attn_return, dim=0)
                    attn_return = torch.mean(attn_return, dim=0) # 返回一个[16384, 4],也就是最后一个序列的[16384, 4]
                    attn_return = torch.mean(attn_return, dim=1) # 返回一个[16384],也就是最后一个序列的[16384]
            elif self.attn_selet_way == 1:
                if self.sequencelevel:
                    for j in range(self.seq_len):
                        attn_tmp = []
                        for i in range(len(attn_all)):
                            attn_tmp.append(attn_all[i][:,:,2*j,2*j+1] * self.ratio_s2aanda2s + attn_all[i][:,:,2*j+1,2*j+2] * (1 - self.ratio_s2aanda2s))
                        attn_tmp = torch.stack(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=1)
                        attn_return.append(attn_tmp) #返回5个[16384, 4]，也就是每个时刻一个[16384, 4]，4表示注意力头数
                    attn_return = torch.stack(attn_return, dim=1)
                else:
                    for i in range(len(attn_all)):
                        attn_return.append(attn_all[i][:,:,2*self.seq_len-2, 2*self.seq_len-1] * self.ratio_s2aanda2s + \
                                        attn_all[i][:,:,2*self.seq_len-1, 2*self.seq_len] * (1 - self.ratio_s2aanda2s))
                    attn_return = torch.stack(attn_return, dim=0)
                    attn_return = torch.mean(attn_return, dim=0) # 返回一个[16384, 4],也就是最后一个序列的[16384, 4]
                    attn_return = torch.mean(attn_return, dim=1) # 返回一个[16384],也就是最后一个序列的[16384]
            else:
                if self.sequencelevel:
                    for j in range(self.seq_len):
                        attn_tmp = []
                        for i in range(len(attn_all)):
                            attn_tmp.append(attn_all[i][:,:,2*j+1,2*j+2])
                        attn_tmp = torch.stack(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=0)
                        attn_tmp = torch.mean(attn_tmp, dim=1)
                        attn_return.append(attn_tmp) #返回5个[16384, 4]，也就是每个时刻一个[16384, 4]，4表示注意力头数
                    attn_return = torch.stack(attn_return, dim=1)
                else:
                    for i in range(len(attn_all)):
                        attn_return.append(attn_all[i][:,:,2*self.seq_len-1,2*self.seq_len])
                    attn_return = torch.stack(attn_return, dim=0)
                    attn_return = torch.mean(attn_return, dim=0) # 返回一个[16384, 4],也就是最后一个序列的[16384, 4]
                    attn_return = torch.mean(attn_return, dim=1) # 返回一个[16384],也就是最后一个序列的[16384]

        # print(attn_return.shape)

        intput_cat = self.norm(intput_cat) # x是[1024, 11, 128]

        # return x, mask, ids_restore
        return intput_cat, attn_return

    def forward_decoder(self, x):
        """
        Decoder processing adjusted for direct input without masking.
        """
        # 将输入进行拼接obs和action拼接到一起,第一个是cls,所应该忽略第一个拼接第二个,
        x = self.decoder_embed(x) # 输入x是[1024, 6, 128]

        if self.use_cls:
            # add pos embed to non-cls-tokens
            x[:, 1:, :] = x[:, 1:, :] + self.decoder_pos_embed_after(x[:, 1:, :]) # 输出x是[1024, 16, 64]
        else:
            # add pos embed to all tokens
            x = x + self.decoder_pos_embed_after(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, attn = blk(x)
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
        z, summed_attn = self.forward_encoder(x, seq_act, seq_next_obs) # z:[16384, 12, 128]
        # 处理拼接问题
        # ----------------------------------------------------------------
        if self.use_cls:
            merged_list = [z[:, 0:1, :]]  # 保留CLS标记, merged_list:[16384, 1, 64]
            for i in range(1, z.size(1) - 1, 2):  # 从1开始，每次跳过2，忽略最后一个如果是奇数
                # 拼接相邻的输入对
                pair = torch.cat([z[:, i:i+1, :], z[:, i+1:i+2, :]], dim=2) # pair:[16384, 1, 128]
                # 使用定义好的线性层处理拼接的向量
                merged = self.linear_merge(pair) # merged:[16384, 1, 64]
                merged_list.append(merged)
        else:
            merged_list = []
            for i in range(0, z.size(1) - 1, 2):  # 从1开始，每次跳过2，忽略最后一个如果是奇数
                # 拼接相邻的输入对
                pair = torch.cat([z[:, i:i+1, :], z[:, i+1:i+2, :]], dim=2)
                # 使用定义好的线性层处理拼接的向量
                merged = self.linear_merge(pair)
                merged_list.append(merged)

        # 将处理后的列表再次合并为tensor
        merged_z = torch.cat(merged_list, dim=1)
        # ----------------------------------------------------------------
        pred = self.forward_decoder(merged_z)
        loss = self.forward_loss(seq_next_obs, pred, keep_batch)

        # Your loss calculation logic might need adjustment based on the task.
        # For simplicity, it's not included here.
        summed_attn = summed_attn * self.attn_ratio_weight

        return loss, summed_attn # 要么是[16384],要么是[16384, 5]

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



