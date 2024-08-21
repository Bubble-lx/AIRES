import gym
from typing import Dict, Any
import torch
import numpy as np
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.common_func import normalize_rewards
from src.utils.enum_types import NormType
from src.utils.running_mean_std import RunningMeanStd
from src.algo.mimex.utils import *
from src.algo.intrinsic_rewards.bert import *

# from stable_baselines3.common.utils import get_device

class MIMExModel(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        # Method-specific params
        rnd_err_norm: int = 0,
        rnd_err_momentum: float = -1.0,
        rnd_use_policy_emb: int = 1,
        policy_cnn: Type[nn.Module] = None,
        policy_rnns: Type[nn.Module] = None,

        # ratio:
        use_my_ratio: bool = False,
        attn_selet_way: int = 0,
        sequencelevel: bool = False,
        ratio_s2aanda2s: float = 0.5,
        attn_ratio_weight: float = 1.0,

        seq_expl_len_ratio: int = 5,
        decoder_depth_ratio: int = 1,
        use_cls_ratio: bool = True,
        norm_loss_ratio: bool = False,
        decoder_num_heads_ratio: int = 1,
        bert_lr_ratio: float = 1e-4,
        # mimex
        seq_expl_len_mimex: int = 5,
        decoder_depth_mimex: int = 1,
        mask_ratio_mimex: float = 0.7,
        use_cls_mimex: bool = True,
        norm_loss_mimex: bool = False,
        k_expl_mimex: float = 0.5,
        n_mask_mimex: int = 1,
        bert_lr_mimex: float = 1e-4,
        decoder_num_heads_mimex: float = 2,
        anneal_k_mimex: bool = False,

    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)

        self.policy_cnn = policy_cnn
        self.policy_rnns = policy_rnns
        self.rnd_use_policy_emb = rnd_use_policy_emb
        self.rnd_err_norm = rnd_err_norm
        self.rnd_err_momentum = rnd_err_momentum
        self.rnd_err_running_stats = RunningMeanStd(momentum=self.rnd_err_momentum)
        self.noveld_visited_obs = dict()  # A dict of set

        # The following constant values are from the original paper:
        # Results show that α = 0.5 and β = 0 works the best (page 7)
        self.noveld_alpha = 0.5
        self.noveld_beta = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.use_my_ratio = use_my_ratio
        self.attn_selet_way = attn_selet_way
        self.sequencelevel = sequencelevel
        self.ratio_s2aanda2s = ratio_s2aanda2s
        self.attn_ratio_weight = attn_ratio_weight

        self.seq_expl_len_ratio = seq_expl_len_ratio
        self.decoder_depth_ratio = decoder_depth_ratio
        self.norm_loss_ratio = norm_loss_ratio
        self.use_cls_ratio = use_cls_ratio
        self.decoder_num_heads_ratio = decoder_num_heads_ratio
        self.bert_lr_ratio = bert_lr_ratio

        # mimex:
        self.seq_expl_len_mimex = seq_expl_len_mimex
        self.decoder_depth_mimex = decoder_depth_mimex
        self.mask_ratio_mimex = mask_ratio_mimex
        self.use_cls_mimex = use_cls_mimex
        self.norm_loss_mimex = norm_loss_mimex
        self.k_expl_mimex = k_expl_mimex
        self.n_mask_mimex = n_mask_mimex
        self.bert_lr_mimex = bert_lr_mimex
        self.decoder_num_heads_mimex = decoder_num_heads_mimex
        self.anneal_k_mimex = anneal_k_mimex

        self._build()



    def _build(self) -> None:
        # Build CNN and RNN
        super()._build()
        self.model_mimex = BERT(
                        seq_len=self.seq_expl_len_mimex,
                        feature_dim=input_dim_mimex,
                        embed_dim=decoder_embed_dim_mimex,
                        decoder_embed_dim=decoder_embed_dim_mimex,
                        decoder_num_heads=self.decoder_num_heads_mimex,
                        decoder_depth=self.decoder_depth_mimex,
                        mask_ratio=self.mask_ratio_mimex,
                        norm_loss=self.norm_loss_mimex,
                        use_cls=self.use_cls_mimex,
                        ).to(self.device)
        self.bert_ratio_opt = torch.optim.Adam(
                    self.model_mimex.parameters(), lr=self.bert_lr_mimex)

    def optimize(self, rollout_data, stats_logger):
        pass

    def get_intrinsic_rewards(self, seq_obs):
        self.seq_obs = seq_obs
        # M, N, T, D = self.seq_obs.shape
        # bert_input = self.seq_obs.view(M * N, T, D) # self.seq_obs:[32, 512, 5, 384], bert_input:[16384, 5, 384]
        # bert_loss, _, mask, loss_reward = self.model_mimex(bert_input, keep_batch=True)
        bert_loss, _, mask, loss_reward = self.model_mimex(seq_obs, keep_batch=True)
        if self.n_mask_mimex > 1:
            # assert self.n_mask_mimex <= 1
            # mask multiple times and calculate average to reduce variance
            for _ in range(self.n_mask_mimex - 1):
                # l, _, _ = self.model_mimex(bert_input, keep_batch=True)
                l, _, _ = self.model_mimex(seq_obs, keep_batch=True)
                bert_loss += l
            bert_loss /= self.n_mask_mimex

        # loss_reward = (loss_reward * mask).sum(dim=-1) / mask.sum(dim=-1)
        loss_reward = bert_loss

        # self.bert_loss = bert_loss
        self.bert_ratio_opt.zero_grad(set_to_none=True)
        (bert_loss.mean()).backward()
        self.bert_ratio_opt.step()

        return bert_loss.detach(), mask.detach(), loss_reward.detach()
