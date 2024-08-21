# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.optim.optim_factory as optim_factory
import os
import utils

from bert import BERT
# ----------------------------------------------------------------
from bert_ratio import BERT_RATIO
from bert_ratio_mask_feat import BERT_RATIO_MASK_FEAT
from bert_ratio_mask_seq import BERT_RATIO_MASK
# ----------------------------------------------------------------
from icm import ICM
from rnd import RND
from ngu import NGU

from mae import MAE, get_data_aug
from mae_models import mae_vit_mini_flex, vit_mini_flex


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder_obs_feat(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())
        self.repr_dim = feature_dim

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return self.trunk(h)

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 encoder_cfg, expl_cfg):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.expl_cfg = expl_cfg

        # models
        if encoder_cfg.aug_type == 'drq':
            self.aug = RandomShiftsAug(pad=4)
        elif encoder_cfg.aug_type == 'mae':
            self.aug = get_data_aug(out_size=obs_shape[-1])
        else:
            raise ValueError(f'Aug type [{self.encoder_cfg.aug_type}] not supported!')

        self.encoder_cfg = encoder_cfg
        if self.encoder_cfg.encoder_type == 'drq':
            self.encoder = Encoder(obs_shape).to(device)
        elif self.encoder_cfg.encoder_type == 'vit':
            # train with PG only
            self.encoder = vit_mini_flex(
                img_size=obs_shape[-1], in_chans=obs_shape[0],
                patch_size=self.encoder_cfg.patch_size
                ).to(device)
        elif self.encoder_cfg.encoder_type == 'mae':
            # train with PG + SSL
            self.encoder = mae_vit_mini_flex(
                img_size=obs_shape[-1], in_chans=obs_shape[0],
                patch_size=self.encoder_cfg.patch_size
                ).to(device)
            # lr = blr * batch_size / 256
            lr = self.encoder_cfg.base_lr * self.encoder_cfg.batch_size / 256
            param_groups = optim_factory.add_weight_decay(
                self.encoder, self.encoder_cfg.weight_decay)
            self.encoder_ssl_opt = torch.optim.AdamW(
                param_groups, lr=lr, betas=(0.9, 0.95))
        else:
            raise ValueError(f'Encoder type [{self.encoder_cfg.encoder_type}] not supported!')

        if self.expl_cfg.seq_expl_len > 0:
            if self.expl_cfg.use_ema_encoder:
                # set up EMA encoder
                self.ema_encoder = Encoder(obs_shape).to(device)
                self.ema_encoder.load_state_dict(self.encoder.state_dict())
                self.ema_encoder.train()

            if self.expl_cfg.use_actor_feat:
                bert_input_feature_dim = feature_dim
            else:
                bert_input_feature_dim = self.encoder.repr_dim

            if self.expl_cfg.baseline == 'icm':
                self.icm = ICM(
                    bert_input_feature_dim, action_shape[-1]).to(device)
            elif self.expl_cfg.baseline == 'rnd':
                self.rnd = RND(bert_input_feature_dim).to(device)
            elif self.expl_cfg.baseline == 'ngu':
                self.ngu = NGU(bert_input_feature_dim, action_shape[-1]).to(device)
            else:
                self.bert = BERT(
                    seq_len=self.expl_cfg.seq_expl_len,
                    feature_dim=bert_input_feature_dim,
                    embed_dim=self.expl_cfg.embed_dim,
                    decoder_embed_dim=self.expl_cfg.decoder_embed_dim,
                    decoder_num_heads=self.expl_cfg.decoder_num_heads,
                    decoder_depth=self.expl_cfg.decoder_depth,
                    mask_ratio=self.expl_cfg.mask_ratio,
                    norm_loss=self.expl_cfg.norm_loss,
                    use_cls=self.expl_cfg.use_cls
                    ).to(device)
                self.bert_opt = torch.optim.Adam(
                    self.bert.parameters(), lr=self.expl_cfg.bert_lr)
        #  ----------------------------------------------------------------
        # print(action_shape)
        self.use_my_ratio = self.expl_cfg.use_my_ratio
        if self.expl_cfg.const_repalce_ratio != 0:
            self.use_my_ratio = False
        if self.use_my_ratio:
            if self.expl_cfg.use_self_encoder:
                self.encoder_obs_feat = Encoder_obs_feat(obs_shape, 100).to(device)
                # path = "Download/premier-taco/pretrained_ckpt/encoder.pt"
                current_file_path = os.path.abspath(__file__)
                current_dir = os.path.dirname(current_file_path)
                # parent_dir_of_parent = os.path.dirname(current_dir)
                current_dir = current_dir + "/pretrained/encoder.pt"
                self.encoder_obs_feat.load_state_dict(torch.load(current_dir))
                if self.expl_cfg.use_mask_ratio:
                    if self.expl_cfg.feat_mask:
                        self.bert_ratio = BERT_RATIO_MASK_FEAT(
                            seq_len=self.expl_cfg.expl_seq_len_ratio,
                            # feature_dim=bert_input_feature_dim,
                            feature_dim=self.encoder_obs_feat.repr_dim,
                            embed_dim=self.expl_cfg.embed_dim_ratio,

                            depth=self.expl_cfg.encoder_depth_ratio,
                            num_heads=self.expl_cfg.encoder_num_heads_ratio,
                            use_next_obs=self.expl_cfg.use_next_obs,

                            decoder_embed_dim=self.expl_cfg.decoder_embed_dim,
                            decoder_num_heads=self.expl_cfg.decoder_num_heads_ratio,
                            decoder_depth=self.expl_cfg.decoder_depth_ratio,
                            mask_ratio=self.expl_cfg.feat_mask_value,
                            norm_loss=self.expl_cfg.norm_loss_ratio,
                            use_cls=self.expl_cfg.use_cls_ratio,
                            action_shape=action_shape[-1],
                            attn_selet_way=expl_cfg.attn_selet_way,
                            ratio_s2aanda2s=expl_cfg.ratio_s2aanda2s,
                            attn_speed_way = expl_cfg.attn_speed_way,
                            attn_ratio_weight=expl_cfg.attn_ratio_weight,
                            ).to(device)
                    else:
                        self.bert_ratio = BERT_RATIO_MASK(
                            seq_len=self.expl_cfg.expl_seq_len_ratio,
                            # feature_dim=bert_input_feature_dim,
                            feature_dim=self.encoder_obs_feat.repr_dim,
                            embed_dim=self.expl_cfg.embed_dim_ratio,

                            depth=self.expl_cfg.encoder_depth_ratio,
                            num_heads=self.expl_cfg.encoder_num_heads_ratio,
                            use_next_obs=self.expl_cfg.use_next_obs,

                            decoder_embed_dim=self.expl_cfg.decoder_embed_dim,
                            decoder_num_heads=self.expl_cfg.decoder_num_heads_ratio,
                            decoder_depth=self.expl_cfg.decoder_depth_ratio,
                            mask_ratio=self.expl_cfg.seq_mask_value,
                            norm_loss=self.expl_cfg.norm_loss_ratio,
                            use_cls=self.expl_cfg.use_cls_ratio,
                            action_shape=action_shape[-1],
                            attn_selet_way=expl_cfg.attn_selet_way,
                            ratio_s2aanda2s=expl_cfg.ratio_s2aanda2s,
                            attn_speed_way = expl_cfg.attn_speed_way,
                            attn_ratio_weight=expl_cfg.attn_ratio_weight,
                            ).to(device)
                else:
                    self.bert_ratio = BERT_RATIO(
                                seq_len=self.expl_cfg.expl_seq_len_ratio,
                                # feature_dim=bert_input_feature_dim,
                                feature_dim=self.encoder_obs_feat.repr_dim,
                                embed_dim=self.expl_cfg.embed_dim_ratio,

                                depth=self.expl_cfg.encoder_depth_ratio,
                                num_heads=self.expl_cfg.encoder_num_heads_ratio,
                                use_next_obs=self.expl_cfg.use_next_obs,

                                decoder_embed_dim=self.expl_cfg.decoder_embed_dim,
                                decoder_num_heads=self.expl_cfg.decoder_num_heads_ratio,
                                decoder_depth=self.expl_cfg.decoder_depth_ratio,
                                mask_ratio=self.expl_cfg.mask_ratio,
                                norm_loss=self.expl_cfg.norm_loss_ratio,
                                use_cls=self.expl_cfg.use_cls_ratio,
                                action_shape=action_shape[-1],
                                attn_selet_way=expl_cfg.attn_selet_way,
                                ratio_s2aanda2s=expl_cfg.ratio_s2aanda2s,
                                attn_ratio_weight=expl_cfg.attn_ratio_weight,
                                attn_speed_way = expl_cfg.attn_speed_way,
                                ).to(device)
            else:
                if self.expl_cfg.use_mask_ratio:
                    if self.expl_cfg.feat_mask:
                        self.bert_ratio = BERT_RATIO_MASK_FEAT(
                            seq_len=self.expl_cfg.expl_seq_len_ratio,
                            feature_dim=bert_input_feature_dim,
                            # feature_dim=self.encoder_obs_feat.repr_dim,

                            depth=self.expl_cfg.encoder_depth_ratio,
                            num_heads=self.expl_cfg.encoder_num_heads_ratio,
                            use_next_obs=self.expl_cfg.use_next_obs,

                            embed_dim=self.expl_cfg.embed_dim_ratio,
                            decoder_embed_dim=self.expl_cfg.decoder_embed_dim,
                            decoder_num_heads=self.expl_cfg.decoder_num_heads_ratio,
                            decoder_depth=self.expl_cfg.decoder_depth_ratio,
                            mask_ratio=self.expl_cfg.feat_mask_value,
                            norm_loss=self.expl_cfg.norm_loss_ratio,
                            use_cls=self.expl_cfg.use_cls_ratio,
                            action_shape=action_shape[-1],
                            attn_selet_way=expl_cfg.attn_selet_way,
                            ratio_s2aanda2s=expl_cfg.ratio_s2aanda2s,
                            attn_speed_way = expl_cfg.attn_speed_way,
                            attn_ratio_weight=expl_cfg.attn_ratio_weight,
                            ).to(device)
                    else:
                        self.bert_ratio = BERT_RATIO_MASK(
                            seq_len=self.expl_cfg.expl_seq_len_ratio,
                            feature_dim=bert_input_feature_dim,
                            # feature_dim=self.encoder_obs_feat.repr_dim,

                            depth=self.expl_cfg.encoder_num_heads_ratio,
                            num_heads=self.expl_cfg.encoder_depth_ratio,
                            use_next_obs=self.expl_cfg.use_next_obs,

                            embed_dim=self.expl_cfg.embed_dim_ratio,
                            decoder_embed_dim=self.expl_cfg.decoder_embed_dim,
                            decoder_num_heads=self.expl_cfg.decoder_num_heads_ratio,
                            decoder_depth=self.expl_cfg.decoder_depth_ratio,
                            mask_ratio=self.expl_cfg.seq_mask_value,
                            norm_loss=self.expl_cfg.norm_loss_ratio,
                            use_cls=self.expl_cfg.use_cls_ratio,
                            action_shape=action_shape[-1],
                            attn_selet_way=expl_cfg.attn_selet_way,
                            ratio_s2aanda2s=expl_cfg.ratio_s2aanda2s,
                            attn_ratio_weight=expl_cfg.attn_ratio_weight,
                            attn_speed_way = expl_cfg.attn_speed_way,
                            ).to(device)
                else:
                    self.bert_ratio = BERT_RATIO(
                        seq_len=self.expl_cfg.expl_seq_len_ratio,
                        feature_dim=bert_input_feature_dim,
                        # feature_dim=self.encoder_obs_feat.repr_dim,

                        depth=self.expl_cfg.encoder_num_heads_ratio,
                        num_heads=self.expl_cfg.encoder_depth_ratio,
                        use_next_obs=self.expl_cfg.use_next_obs,

                        embed_dim=self.expl_cfg.embed_dim_ratio,
                        decoder_embed_dim=self.expl_cfg.decoder_embed_dim,
                        decoder_num_heads=self.expl_cfg.decoder_num_heads_ratio,
                        decoder_depth=self.expl_cfg.decoder_depth_ratio,
                        mask_ratio=self.expl_cfg.mask_ratio,
                        norm_loss=self.expl_cfg.norm_loss_ratio,
                        use_cls=self.expl_cfg.use_cls_ratio,
                        action_shape=action_shape[-1],
                        attn_selet_way=expl_cfg.attn_selet_way,
                        ratio_s2aanda2s=expl_cfg.ratio_s2aanda2s,
                        attn_ratio_weight=expl_cfg.attn_ratio_weight,
                        attn_speed_way = expl_cfg.attn_speed_way,
                        ).to(device)
            self.bert_ratio_opt = torch.optim.Adam(
                        self.bert_ratio.parameters(), lr=self.expl_cfg.bert_lr)
        #  ----------------------------------------------------------------

        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.episodic_obs_emb_history = [None for _ in range(self.encoder_cfg.batch_size)]

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.attn_all = 0

        self.train()
        self.critic_target.train()


        # self.firts_time_tmp = 1

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        '''
        obs: (frame_stack * C, H, W)
        '''
        obs = torch.as_tensor(obs, device=self.device) # [9,84,84]
        obs = self.encoder(obs.unsqueeze(0).float()) # [1,39200]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def get_seq_expl_r(self, seq_obs):
        if self.expl_cfg.use_ema_encoder:
            # update EMA encoder weights
            utils.soft_update_params(
                self.encoder, self.ema_encoder, self.expl_cfg.ema_tau)
            encoder = self.ema_encoder
        else:
            encoder = self.encoder

        n, t, c, h, w = seq_obs.size() # n:1024, t:5, c:9, h:84, w:84, seq_obs:[512, 5, 9, 84, 84]
        seq_obs = self.aug(
            seq_obs.float().view(n, t*c, h, w)).view(n, t, c, h, w)
        with torch.no_grad():
            if self.expl_cfg.use_actor_feat:
                seq_emb = encoder(seq_obs.view(n * t, c, h, w))
                seq_emb = self.actor.trunk(seq_emb).view(n, t, -1)
            else:
                seq_emb = encoder(seq_obs.view(n * t, c, h, w)).view(
                    n, t, -1) # seq_emb:[1024, 5, 39200]
        bert_loss, _, mask, loss_reward = self.bert(seq_emb.detach(), keep_batch=True)

        # optimize BERT
        self.bert_opt.zero_grad(set_to_none=True)
        (bert_loss.mean()).backward()
        self.bert_opt.step()

        return bert_loss

    def get_expl_ratio(self, seq_obs, seq_act, seq_next_obs):
        if self.expl_cfg.use_ema_encoder:
            # update EMA encoder weights
            utils.soft_update_params(
                self.encoder, self.ema_encoder, self.expl_cfg.ema_tau)
            encoder = self.ema_encoder
        elif self.expl_cfg.use_self_encoder:
            encoder = self.encoder_obs_feat
        else:
            encoder = self.encoder
        n, t, c, h, w = seq_obs.size()
        with torch.no_grad():
            if self.expl_cfg.use_actor_feat: # 默认false
                seq_emb = encoder(seq_obs.view(n * t, c, h, w))
                seq_emb = self.actor.trunk(seq_emb).view(n, t, -1)
            else:
                seq_emb = encoder(seq_obs.view(n * t, c, h, w)).view(
                    n, t, -1) # [1024, 5, 39200]
                seq_next_obs_emb = encoder(seq_next_obs.view(n * t, c, h, w)).view(
                    n, t, -1) # [1024, 5, 39200]

        bert_loss, summed_attn, self.attn_all = self.bert_ratio(seq_emb.detach(), seq_act, seq_next_obs_emb.detach(), keep_batch=True)

        # optimize BERT
        self.bert_ratio_opt.zero_grad(set_to_none=True)
        (bert_loss.mean()).backward()
        self.bert_ratio_opt.step()
        ratio = summed_attn

        # return bert loss (N,) as exploration reward
        return ratio

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        

        if self.expl_cfg.seq_expl_len > 0:
            if self.expl_cfg.baseline == 'icm':
                obs, action, reward, discount, next_obs, seq_obs, seq_act, seq_next_obs, seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio = utils.to_torch(
                    batch, self.device)

                # compute intrinsic reward using ICM
                obs_emb = self.encoder(self.aug(obs.float())) # obs_emb is [512, 39200]
                next_obs_emb = self.encoder(self.aug(next_obs.float()))

                # update inverse dynamics model and get features
                # predict next_obs using forward dynamics model and update model
                icm_loss = self.icm(obs_emb, next_obs_emb, action) # icm_loss:[512]
                expl_r = icm_loss.view(*reward.shape) # expl_r[512, 1]
                if self.use_my_ratio:
                    bert_ratio = self.get_expl_ratio(seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio) # bert_ratio: [512, 5]
                    bert_ratio = bert_ratio.view(*reward.shape).detach() # bert_ratio: [512],reward.shape [512, 1]
                else:
                    bert_ratio = 1
                if self.expl_cfg.const_repalce_ratio != 0:
                    bert_ratio = self.expl_cfg.const_repalce_ratio
                # relabel rewards
                expl_r = expl_r.detach() * bert_ratio

            elif self.expl_cfg.baseline == 'ngu':
                obs, action, reward, discount, next_obs, seq_obs, seq_act, seq_next_obs, seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio = utils.to_torch(
                    batch, self.device)
                obs_emb = self.encoder(self.aug(obs.float())) # obs_emb is [512, 39200]
                next_obs_emb = self.encoder(self.aug(next_obs.float()))
                self.episodic_obs_emb_history = [None for _ in range(self.encoder_cfg.batch_size)]
                ngu_loss = self.ngu(obs_emb, next_obs_emb, action, self.episodic_obs_emb_history) # icm_loss:[512]
                ngu_loss = torch.from_numpy(ngu_loss).float().to(self.device)
                expl_r = ngu_loss.view(*reward.shape) # expl_r[512, 1]
                if self.use_my_ratio:
                    bert_ratio = self.get_expl_ratio(seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio) # bert_ratio: [512, 5]
                    bert_ratio = bert_ratio.view(*reward.shape).detach()
                else:
                    bert_ratio = 1

                if self.expl_cfg.const_repalce_ratio != 0:
                    bert_ratio = self.expl_cfg.const_repalce_ratio

                # relabel rewards
                expl_r = expl_r.detach() * bert_ratio

            elif self.expl_cfg.baseline == 'rnd':
                obs, action, reward, discount, next_obs, seq_obs, seq_act, seq_next_obs, seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio = utils.to_torch(
                    batch, self.device)

                # compute intrinsic reward using RND
                obs_emb = self.encoder(self.aug(obs.float()))
                rnd_loss = self.rnd(obs_emb)
                expl_r = rnd_loss.view(*reward.shape)
                if self.use_my_ratio:
                    bert_ratio = self.get_expl_ratio(seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio) # bert_ratio: [512, 5]
                    bert_ratio = bert_ratio.view(*reward.shape).detach()
                else:
                    bert_ratio = 1
                if self.expl_cfg.const_repalce_ratio != 0:
                    bert_ratio = self.expl_cfg.const_repalce_ratio
                expl_r = expl_r.detach() * bert_ratio
            else:

                obs, action, reward, discount, next_obs, seq_obs, seq_act, seq_next_obs, seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio = utils.to_torch(
                    batch, self.device)

                if self.use_my_ratio:
                    bert_ratio = self.get_expl_ratio(seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio) # bert_ratio: [512, 5]
                else:
                    bert_ratio = 1

                expl_r = self.get_seq_expl_r(seq_obs).view(*reward.shape) # [1024, 5, 9, 84, 84]
                if self.expl_cfg.n_masks > 1:
                    # mask multiple times
                    assert self.n_mask <= 1
                    n_masks = self.expl_cfg.n_masks - 1
                    for _ in range(n_masks):
                        expl_r += self.get_seq_expl_r(seq_obs).view(*reward.shape)
                    expl_r /= n_masks
                if self.use_my_ratio:
                    bert_ratio = bert_ratio.view(*reward.shape).detach()
                else:
                    bert_ratio = 1
                if self.expl_cfg.const_repalce_ratio != 0:
                    bert_ratio = self.expl_cfg.const_repalce_ratio
                expl_r = expl_r.detach() * bert_ratio

            if self.expl_cfg.save_ratio:
                if self.use_my_ratio:
                    metrics['ratio'] = bert_ratio.mean().item()
                else:
                    metrics['ratio'] = bert_ratio

            if self.use_tb:
                metrics['expl_reward'] = expl_r.mean().item()
            reward += expl_r * self.expl_cfg.k_expl
        else:
            obs, action, reward, discount, next_obs, seq_obs, seq_act, seq_next_obs, seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio = utils.to_torch(
                    batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float()) # [1024, 9, 84]

        # encode
        if self.encoder_cfg.encoder_type == 'mae':
            loss, _, _ = self.encoder.update(torch.cat((obs, next_obs), dim=0))
            self.encoder_ssl_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.encoder_ssl_opt.step()

            if self.use_tb:
                metrics['encoder_loss'] = loss.item()

        obs = self.encoder(obs) # [1024, 39200]
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
