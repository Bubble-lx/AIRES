import gym
import time
import wandb
import numpy as np
import torch as th
from PIL import Image
from matplotlib import pyplot as plt
from gym_minigrid.minigrid import Key, Door, Goal
from src.algo.buffers.ppo_buffer import PPORolloutBuffer
from src.utils.loggers import StatisticsLogger, LocalLogger
from src.utils.common_func import set_random_seed
from src.utils.enum_types import ModelType, EnvSrc
from src.algo.mimex.utils import draw_heat_map
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from gym import spaces
from torch.cuda.amp import autocast, GradScaler
from typing import Any, Dict, Optional, Tuple, Type, Union
from src.algo.mimex.utils import *
from src.algo.mimex.bert_ratio import BERT_RATIO
from algo.mimex.bert_ratio_mask_seq import BERT_RATIO_MASK
from algo.mimex.bert_ratio_mask_feat import BERT_RATIO_MASK_FEAT

class PPORollout(BaseAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        run_id: int,
        learning_rate: Union[float, Schedule],
        n_steps: int,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        pg_coef: float,
        vf_coef: float,
        int_rew_source: ModelType,
        int_rew_coef: float,
        int_rew_norm : int,
        int_rew_momentum: Optional[float],
        int_rew_eps : float,
        int_rew_clip : float,
        adv_momentum : float,
        image_noise_scale : float,
        enable_plotting : int,
        can_see_walls : int,
        ext_rew_coef: float,
        adv_norm: int,
        adv_eps: float,
        max_grad_norm: float,
        use_sde: bool,
        game_name: str,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        env_source: Optional[EnvSrc] = None,
        env_render: Optional[int] = None,
        fixed_seed: Optional[int] = None,
        plot_interval: int = 10,
        plot_colormap: str = "Blues",
        log_explored_states: Optional[int] = None,
        local_logger: Optional[LocalLogger] = None,
        use_wandb: bool = False,

        use_self_encoder: bool = False,
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
        encoder_num_heads_ratio: int = 4,
        encoder_depth_ratio: int = 4,
        use_next_obs: bool = True,
        const_repalce_ratio: float = 0.25,
        attn_speed_way: int = 0,

        use_mask_ratio: bool = False,
        seq_mask_value: float = 0.4,
        feat_mask: bool = False,
        feat_mask_value: float = 0.4,

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
        super(PPORollout, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
        )
        self.run_id = run_id
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.pg_coef = pg_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_timesteps = 0
        self.int_rew_source = int_rew_source
        self.int_rew_coef = int_rew_coef
        self.int_rew_norm = int_rew_norm
        self.int_rew_eps = int_rew_eps
        self.adv_momentum = adv_momentum
        self.int_rew_clip = int_rew_clip
        self.image_noise_scale = image_noise_scale
        self.enable_plotting = enable_plotting
        self.can_see_walls = can_see_walls
        self.int_rew_momentum = int_rew_momentum
        self.ext_rew_coef = ext_rew_coef
        self.adv_norm = adv_norm
        self.adv_eps = adv_eps
        self.env_source = env_source
        self.env_render = env_render
        self.fixed_seed = fixed_seed
        self.plot_interval = plot_interval
        self.plot_colormap = plot_colormap
        self.log_explored_states = log_explored_states
        self.local_logger = local_logger
        self.use_wandb = use_wandb
        self.game_name = game_name
        self.first_time = 0

        self.use_self_encoder = use_self_encoder
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
        self.encoder_num_heads_ratio = encoder_num_heads_ratio
        self.encoder_depth_ratio = encoder_depth_ratio
        self.attn_speed_way = attn_speed_way
        self.use_mask_ratio=use_mask_ratio
        self.seq_mask_value=seq_mask_value
        self.feat_mask=feat_mask
        self.feat_mask_value=feat_mask_value

        self.use_next_obs = use_next_obs
        self.const_repalce_ratio = const_repalce_ratio
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

        if self.use_self_encoder:
            self.inputdim = 16
        else:
            self.inputdim = 64 # 64 or 147
        print("self.device is :", self.device)
        if self.int_rew_source == ModelType.MIMEx:
            self.seq_obs_buffer_mimex = th.zeros(
                env.num_envs,
                self.seq_expl_len_mimex,
                self.inputdim,
                dtype=th.float,
                device=self.device)
            self.seq_next_obs_buffer_mimex = th.zeros(
                env.num_envs,
                self.seq_expl_len_mimex,
                self.inputdim,
                dtype=th.float,
                device=self.device)
            # print("self.action_space.shape is :", self.action_space.shape)
            self.seq_act_buffer_mimex = th.zeros(
                env.num_envs,
                self.seq_expl_len_mimex,
                *self.action_space.shape,
                dtype=th.float,
                device=self.device)

        if isinstance(self.action_space, spaces.Discrete):
            action_shape_local = 1
        elif isinstance(self.action_space, spaces.Box):
            action_shape_local = self.action_space.shape[0]
        else:
            raise TypeError("Unsupported action space type.")
        if self.use_my_ratio:
            self.seq_obs_buffer_ratio = th.zeros(
                env.num_envs,
                self.seq_expl_len_ratio,
                self.inputdim,
                dtype=th.float,
                device=self.device)
            self.seq_next_obs_buffer_ratio = th.zeros(
                env.num_envs,
                self.seq_expl_len_ratio,
                self.inputdim,
                dtype=th.float,
                device=self.device)
            # print("self.action_space.shape is :", self.action_space.shape)
            self.seq_act_buffer_ratio = th.zeros(
                env.num_envs,
                self.seq_expl_len_ratio,
                action_shape_local,
                dtype=th.float,
                device=self.device)
            if self.use_self_encoder:
                if self.use_mask_ratio:
                    if self.feat_mask:
                        self.bert_ratio = BERT_RATIO_MASK_FEAT(
                        seq_len=self.seq_expl_len_ratio,
                        feature_dim=16,
                        depth=self.encoder_depth_ratio,
                        num_heads=self.encoder_num_heads_ratio,
                        use_next_obs=self.use_next_obs,

                        embed_dim=8,
                        decoder_embed_dim=8,
                        decoder_num_heads=self.decoder_num_heads_ratio,
                        decoder_depth=self.decoder_depth_ratio,
                        mask_ratio=self.feat_mask_value,
                        norm_loss=self.norm_loss_ratio,
                        use_cls=self.use_cls_ratio,
                        action_shape= action_shape_local,
                        attn_selet_way=self.attn_selet_way,
                        ratio_s2aanda2s=self.ratio_s2aanda2s,
                        attn_ratio_weight=self.attn_ratio_weight,
                        attn_speed_way = self.attn_speed_way,
                        ).to(self.device)
                    else:
                        self.bert_ratio = BERT_RATIO_MASK(
                        seq_len=self.seq_expl_len_ratio,
                        feature_dim=16,
                        depth=self.encoder_depth_ratio,
                        num_heads=self.encoder_num_heads_ratio,
                        use_next_obs=self.use_next_obs,

                        embed_dim=8,
                        decoder_embed_dim=8,
                        decoder_num_heads=self.decoder_num_heads_ratio,
                        decoder_depth=self.decoder_depth_ratio,
                        mask_ratio=self.seq_mask_value,
                        norm_loss=self.norm_loss_ratio,
                        use_cls=self.use_cls_ratio,
                        action_shape= action_shape_local,
                        attn_selet_way=self.attn_selet_way,
                        ratio_s2aanda2s=self.ratio_s2aanda2s,
                        attn_ratio_weight=self.attn_ratio_weight,
                        attn_speed_way = self.attn_speed_way,
                        ).to(self.device)
                else:
                    self.bert_ratio = BERT_RATIO(
                    seq_len=self.seq_expl_len_ratio,
                    feature_dim=16,
                    depth=self.encoder_depth_ratio,
                    num_heads=self.encoder_num_heads_ratio,
                    use_next_obs=self.use_next_obs,

                    embed_dim=8,
                    decoder_embed_dim=8,
                    decoder_num_heads=self.decoder_num_heads_ratio,
                    decoder_depth=self.decoder_depth_ratio,
                    mask_ratio=mask_ratio,
                    norm_loss=self.norm_loss_ratio,
                    use_cls=self.use_cls_ratio,
                    action_shape= action_shape_local,
                    attn_selet_way=self.attn_selet_way,
                    ratio_s2aanda2s=self.ratio_s2aanda2s,
                    attn_ratio_weight=self.attn_ratio_weight,
                    attn_speed_way = self.attn_speed_way,
                    ).to(self.device)
            else:
                if self.use_mask_ratio:
                    if self.feat_mask:
                        self.bert_ratio = BERT_RATIO_MASK_FEAT(
                            seq_len=self.seq_expl_len_ratio,
                            feature_dim=input_dim,
                            depth=self.encoder_depth_ratio,
                            num_heads=self.encoder_num_heads_ratio,
                            use_next_obs=self.use_next_obs,

                            embed_dim=8,
                            decoder_embed_dim=8,
                            decoder_num_heads=self.decoder_num_heads_ratio,
                            decoder_depth=self.decoder_depth_ratio,
                            mask_ratio=self.feat_mask_value,
                            norm_loss=self.norm_loss_ratio,
                            use_cls=self.use_cls_ratio,
                            action_shape= action_shape_local,
                            attn_selet_way=self.attn_selet_way,
                            ratio_s2aanda2s=self.ratio_s2aanda2s,
                            attn_ratio_weight=self.attn_ratio_weight,
                            attn_speed_way = self.attn_speed_way,
                            ).to(self.device)
                    else:
                        self.bert_ratio = BERT_RATIO_MASK(
                            seq_len=self.seq_expl_len_ratio,
                            feature_dim=input_dim,
                            depth=self.encoder_depth_ratio,
                            num_heads=self.encoder_num_heads_ratio,
                            use_next_obs=self.use_next_obs,
                            embed_dim=8,
                            decoder_embed_dim=8,
                            decoder_num_heads=self.decoder_num_heads_ratio,
                            decoder_depth=self.decoder_depth_ratio,
                            mask_ratio=self.seq_mask_value,
                            norm_loss=self.norm_loss_ratio,
                            use_cls=self.use_cls_ratio,
                            action_shape= action_shape_local,
                            attn_selet_way=self.attn_selet_way,
                            ratio_s2aanda2s=self.ratio_s2aanda2s,
                            attn_ratio_weight=self.attn_ratio_weight,
                            attn_speed_way = self.attn_speed_way,
                            ).to(self.device)
                else:
                    self.bert_ratio = BERT_RATIO(
                        seq_len=self.seq_expl_len_ratio,
                        feature_dim=input_dim,

                        depth=self.encoder_depth_ratio,
                        num_heads=self.encoder_num_heads_ratio,
                        use_next_obs=self.use_next_obs,

                        embed_dim=decoder_embed_dim,
                        decoder_embed_dim=decoder_embed_dim,
                        decoder_num_heads=self.decoder_num_heads_ratio,
                        decoder_depth=self.decoder_depth_ratio,
                        mask_ratio=mask_ratio,
                        norm_loss=self.norm_loss_ratio,
                        use_cls=self.use_cls_ratio,
                        action_shape= action_shape_local,
                        attn_selet_way=self.attn_selet_way,
                        ratio_s2aanda2s=self.ratio_s2aanda2s,
                        attn_ratio_weight=self.attn_ratio_weight,
                        attn_speed_way = self.attn_speed_way,
                        ).to(self.device)
            self.bert_ratio_opt = th.optim.Adam(
                    self.bert_ratio.parameters(), lr=self.bert_lr_ratio)
        self.record_ratio = True
        if self.record_ratio:
            self.alpha_ratio = []

        self.attn_all = 0

        self.collect_data = False
        if self.collect_data:
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)
            self.parent_dir_of_parent = os.path.dirname(current_dir)
            self.parent_dir_of_parent = self.parent_dir_of_parent + '/dataset_encoder_train/DoorKey-16x16/train'
            if not os.path.exists(self.parent_dir_of_parent):
                os.makedirs(self.parent_dir_of_parent)
            self.img_num = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        set_random_seed(self.seed)
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.ppo_rollout_buffer = PPORolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            features_dim=self.policy.features_dim,
            dim_policy_traj=self.policy.dim_policy_features,
            dim_model_traj=self.policy.dim_model_features,
            int_rew_source=self.int_rew_source,
            int_rew_coef=self.int_rew_coef,
            ext_rew_coef=self.ext_rew_coef,
            int_rew_norm=self.int_rew_norm,
            int_rew_clip=self.int_rew_clip,
            int_rew_eps=self.int_rew_eps,
            adv_momentum=self.adv_momentum,
            adv_norm=self.adv_norm,
            adv_eps=self.adv_eps,
            gru_layers=self.policy.gru_layers,
            int_rew_momentum=self.int_rew_momentum,
            use_status_predictor=self.policy.use_status_predictor,
            use_self_encoder = self.use_self_encoder,
            
            # ratio:
            use_my_ratio = self.use_my_ratio,
            attn_selet_way = self.attn_selet_way,
            sequencelevel = self.sequencelevel,
            ratio_s2aanda2s = self.ratio_s2aanda2s,
            attn_ratio_weight = self.attn_ratio_weight,

            seq_expl_len_ratio = self.seq_expl_len_ratio,
            decoder_depth_ratio = self.decoder_depth_ratio,
            use_cls_ratio = self.use_cls_ratio,
            norm_loss_ratio = self.norm_loss_ratio,
            decoder_num_heads_ratio = self.decoder_num_heads_ratio,
            bert_lr_ratio = self.bert_lr_ratio,
            encoder_num_heads_ratio = self.encoder_num_heads_ratio,
            encoder_depth_ratio = self.encoder_depth_ratio,
            
            # mimex:
            seq_expl_len_mimex = self.seq_expl_len_mimex,
            decoder_depth_mimex = self.decoder_depth_mimex,
            mask_ratio_mimex = self.mask_ratio_mimex,
            use_cls_mimex = self.use_cls_mimex,
            norm_loss_mimex = self.norm_loss_mimex,
            k_expl_mimex = self.k_expl_mimex,
            n_mask_mimex = self.n_mask_mimex,
            bert_lr_mimex = self.bert_lr_mimex,
            decoder_num_heads_mimex = self.decoder_num_heads_mimex,
            anneal_k_mimex = self.anneal_k_mimex,
        )

    def on_training_start(self):
        if isinstance(self._last_obs, Dict):
            self._last_obs = self._last_obs["rgb"]

        if self.env_source == EnvSrc.MiniGrid:
            # Set advanced options for MiniGrid envs
            self.env.can_see_walls = self.can_see_walls
            self.env.image_noise_scale = self.image_noise_scale
            self.env.image_rng = np.random.default_rng(seed=self.run_id + 1313)

            # Initialize seeds for each MiniGrid env
            np.random.seed(self.run_id)
            seeds = np.random.rand(self.n_envs)
            seeds = [int(s * 0x7fffffff) for s in seeds]
            np.random.seed(self.run_id)
            self.env.set_seeds(seeds)
            self.env.waiting = True
            for i in range(self.n_envs):
                self.env.send_reset(env_id=i)
            # for i in range(self.n_envs):
            self.env.set_values(self.image_noise_scale, np.random.default_rng(seed=self.run_id + 1313))
            for i in range(self.n_envs):
                self._last_obs[i] = self.env.recv_obs(env_id=i)
            self.env.waiting = False

            # Init variables for logging
            self.width = self.env.get_attr('width')[0]
            self.height = self.env.get_attr('height')[0]
            self.global_visit_counts = np.zeros([self.width, self.height], dtype=np.int32)
            self.global_reward_map_maxs = np.zeros([self.width, self.height], dtype=np.float64)
            self.global_reward_map_sums = np.zeros([self.width, self.height], dtype=np.float64)
            self.global_reward_map_nums = np.zeros([self.width, self.height], dtype=np.int32)
            self.global_value_map_sums = np.zeros([self.width, self.height], dtype=np.float64)
            self.global_value_map_nums = np.zeros([self.width, self.height], dtype=np.int32)

        self.global_episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.global_episode_intrinsic_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.global_episode_unique_states = np.zeros(self.n_envs, dtype=np.float32)
        self.global_episode_visited_states = [dict() for _ in range(self.n_envs)]
        self.global_lifelong_unique_states = 0
        self.global_lifelong_visited_states = dict()
        self.global_episode_visited_positions = [dict() for _ in range(self.n_envs)]
        self.global_episode_visited_pos_sum = np.zeros(self.n_envs, dtype=np.float32)
        self.global_episode_steps = np.zeros(self.n_envs, dtype=np.int32)

        if self.policy.use_status_predictor:
            self.global_has_keys = np.zeros(self.n_envs, dtype=np.int32)
            self.global_open_doors = np.zeros(self.n_envs, dtype=np.int32)
            self.curr_key_status = np.zeros(self.n_envs, dtype=np.float32)
            self.curr_door_status = np.zeros(self.n_envs, dtype=np.float32)
            self.curr_target_dists = np.zeros((self.n_envs, 3), dtype=np.float32)
        else:
            self.global_has_keys = None
            self.global_open_doors = None
            self.curr_key_status = None
            self.curr_door_status = None
            self.curr_target_dists = None

        self.episodic_obs_emb_history = [None for _ in range(self.n_envs)]
        self.episodic_trj_emb_history = [None for _ in range(self.n_envs)]

        if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
            self.policy.int_rew_model.init_obs_queue(self._last_obs)

        def float_zeros(tensor_shape):
            return th.zeros(tensor_shape, device=self.device, dtype=th.float32)

        self._last_policy_mems = float_zeros([self.n_envs, self.policy.gru_layers, self.policy.dim_policy_features])
        self._last_model_mems = float_zeros([self.n_envs, self.policy.gru_layers, self.policy.dim_model_features])

    def init_on_rollout_start(self):
        # Log statistics data per each rollout
        self.rollout_stats = StatisticsLogger(mode='rollout')
        self.rollout_done_episodes = 0
        self.rollout_done_episode_steps = 0
        self.rollout_sum_rewards = 0
        self.rollout_episode_unique_states = 0
        self.rollout_done_episode_unique_states = 0

    def log_before_transition(self, values):
        if self.env_source == EnvSrc.MiniGrid:
            self._last_state_hash_vals = self.env.env_method('hash')

        # Update Key and Door Status
        agent_positions = None
        if self.policy.use_status_predictor:
            agent_positions = np.array(self.env.get_attr('agent_pos'))
            agent_carryings = self.env.get_attr('carrying')
            env_grids = self.env.get_attr('grid')

            self.curr_door_pos = np.zeros((self.n_envs, 2), dtype=np.int32)
            self.curr_key_pos = np.copy(agent_positions).reshape(self.n_envs, 2)
            self.curr_goal_pos = np.zeros((self.n_envs, 2), dtype=np.int32)
            for env_id in range(self.n_envs):
                # The only possible carrying in DoorKey is the key
                self.global_has_keys[env_id] = int(isinstance(agent_carryings[env_id], Key))

                # Door, Key, Goal positions
                for env_id, grid in enumerate(env_grids[env_id].grid):
                    col = env_id % self.width
                    row = env_id // self.width
                    if isinstance(grid, Door):
                        self.curr_door_pos[env_id] = np.array((col, row))
                        self.global_open_doors[env_id] = int(grid.is_open)
                    elif isinstance(grid, Key):
                        self.curr_key_pos[env_id] = np.array((col, row))
                    elif isinstance(grid, Goal):
                        self.curr_goal_pos[env_id] = np.array((col, row))

            self.curr_key_status = np.copy(self.global_has_keys)
            self.curr_door_status = np.copy(self.global_open_doors)
            self.rollout_stats.add(
                key_status=np.mean(self.global_has_keys),
                door_status=np.mean(self.global_open_doors),
            )

        # Update agent position and visit count
        if self.policy.use_status_predictor or self.enable_plotting:
            if agent_positions is None:
                agent_positions = np.array(self.env.get_attr('agent_pos'))
            for i in range(self.n_envs):
                c, r = agent_positions[i]
                self.global_visit_counts[r, c] += 1
                self.global_value_map_sums[r, c] += values[i].item()
                self.global_value_map_nums[r, c] += 1

            # Current agent position
            self.curr_agent_pos = np.copy(agent_positions)

            # Define the target of position prediction loss
            if self.policy.use_status_predictor:
                # Manhattan Distance to the Door
                key_dists = np.abs(self.curr_agent_pos - self.curr_key_pos)
                key_dists = np.sum(key_dists, axis=1) / (self.width + self.height)
                door_dists = np.abs(self.curr_agent_pos - self.curr_door_pos)
                door_dists = np.sum(door_dists, axis=1) / (self.width + self.height)
                goal_dists = np.abs(self.curr_agent_pos - self.curr_goal_pos)
                goal_dists = np.sum(goal_dists, axis=1) / (self.width + self.height)
                self.curr_target_dists = np.stack([key_dists, door_dists, goal_dists], axis=1)

    def log_after_transition(self, rewards, intrinsic_rewards):
        self.global_episode_rewards += rewards
        self.global_episode_intrinsic_rewards += intrinsic_rewards
        self.global_episode_steps += 1

        # Logging episodic/lifelong visited states, reward map
        if self.log_explored_states:
            # 0 - Not to log
            # 1 - Log both episodic and lifelong states
            # 2 - Log episodic visited states only
            if self.env_source == EnvSrc.MiniGrid:
                agent_positions = np.array(self.env.get_attr('agent_pos'))
                for env_id in range(self.n_envs):
                    c, r = agent_positions[env_id]

                    # count the visited positions
                    pos = c * self.width + r
                    pos_visit_count = self.global_episode_visited_positions[env_id]
                    if pos not in pos_visit_count:
                        pos_visit_count[pos] = 1
                        self.global_episode_visited_pos_sum[env_id] += 1
                    else:
                        pos_visit_count[pos] += 1

                    env_hash = self._last_state_hash_vals[env_id]
                    if env_hash in self.global_episode_visited_states[env_id]:
                        self.global_episode_visited_states[env_id][env_hash] += 1
                    else:
                        self.global_episode_visited_states[env_id][env_hash] = 1
                        self.global_episode_unique_states[env_id] += 1
                        self.rollout_episode_unique_states += 1

                    if self.log_explored_states == 1:
                        if env_hash in self.global_lifelong_visited_states:
                            self.global_lifelong_visited_states[env_hash] += 1
                        else:
                            self.global_lifelong_visited_states[env_hash] = 1
                            self.global_lifelong_unique_states += 1

                    if self.enable_plotting:
                        self.global_reward_map_maxs[r, c] = np.maximum(
                            self.global_reward_map_maxs[r, c],
                            intrinsic_rewards[env_id]
                        )
                        self.global_reward_map_sums[r, c] += intrinsic_rewards[env_id]
                        self.global_reward_map_nums[r, c] += 1

            elif self.env_source == EnvSrc.ProcGen:
                for env_id in range(self.n_envs):
                    # In Procgen games, the counted "states" are observations
                    env_hash = tuple(self._last_obs[env_id].reshape(-1).tolist())

                    if env_hash in self.global_episode_visited_states[env_id]:
                        self.global_episode_visited_states[env_id][env_hash] += 1
                    else:
                        self.global_episode_visited_states[env_id][env_hash] = 1
                        self.global_episode_unique_states[env_id] += 1
                        self.rollout_episode_unique_states += 1

                    if self.log_explored_states == 1:
                        if env_hash in self.global_lifelong_visited_states:
                            self.global_lifelong_visited_states[env_hash] += 1
                        else:
                            self.global_lifelong_visited_states[env_hash] = 1
                            self.global_lifelong_unique_states += 1

    def clear_on_episode_end(self, dones, policy_mems, model_mems):
        for env_id in range(self.n_envs):
            if dones[env_id]:
                if policy_mems is not None: policy_mems[env_id] *= 0.0
                if model_mems is not None: model_mems[env_id] *= 0.0
                self.episodic_obs_emb_history[env_id] = None
                self.episodic_trj_emb_history[env_id] = None
                self.rollout_sum_rewards += self.global_episode_rewards[env_id]
                self.rollout_done_episode_steps += self.global_episode_steps[env_id]
                self.rollout_done_episode_unique_states += self.global_episode_unique_states[env_id]
                self.rollout_done_episodes += 1
                self.global_episode_rewards[env_id] = 0
                self.global_episode_intrinsic_rewards[env_id] = 0
                self.global_episode_unique_states[env_id] = 0
                self.global_episode_visited_states[env_id] = dict()  # logging use
                self.global_episode_visited_positions[env_id] = dict()  # logging use
                self.global_episode_visited_pos_sum[env_id] = 0  # logging use
                self.global_episode_steps[env_id] = 0
                if self.policy.use_status_predictor:
                    self.global_has_keys[env_id] = 0
                    self.global_open_doors[env_id] = 0
                    self.curr_key_status[env_id] = 0
                    self.curr_door_status[env_id] = 0

    def log_on_rollout_end(self, log_interval):
        if log_interval is not None and self.iteration % log_interval == 0:
            log_data = {
                "iterations": self.iteration,
                "time/fps": int(self.num_timesteps / (time.time() - self.start_time)),
                "time/time_elapsed": int(time.time() - self.start_time),
                "time/total_timesteps": self.num_timesteps,
                "rollout/ep_rew_mean": self.rollout_sum_rewards / (self.rollout_done_episodes + 1e-8),
                "rollout/ep_len_mean": self.rollout_done_episode_steps / (self.rollout_done_episodes + 1e-8),
                # unique states / positions
                "rollout/ep_unique_states": self.rollout_done_episode_unique_states / (
                            self.rollout_done_episodes + 1e-8),
                "rollout/ll_unique_states": self.global_lifelong_unique_states,
                "rollout/ep_unique_states_per_step": self.rollout_episode_unique_states / (
                            self.ppo_rollout_buffer.buffer_size * self.n_envs),
                "rollout/ll_unique_states_per_step": self.global_lifelong_unique_states / self.num_timesteps,
                # intrinsic rewards
                "rollout/int_rew_coef": self.ppo_rollout_buffer.int_rew_coef,
                "rollout/int_rew_buffer_mean": self.ppo_rollout_buffer.int_rew_mean,
                "rollout/int_rew_buffer_std": self.ppo_rollout_buffer.int_rew_std,
            }
            recoder_way = 0
            if self.record_ratio:
                if recoder_way == 0:
                    if self.attn_speed_way == 2:
                        log_data.update({
                            "alpha_mean": np.mean(np.mean(self.alpha_ratio, axis=0)),
                        })
                    else:
                        log_data.update({
                            "alpha_mean": np.mean(np.mean(self.alpha_ratio, axis=0)),
                        })
                elif recoder_way == 1:
                    log_data.update({
                        "alpha_mean": np.mean(self.alpha_ratio, axis=1),
                    })
            # record data
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                log_data.update({
                    "rollout/ep_info_rew_mean": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    "rollout/ep_info_len_mean": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                })
            else:
                log_data.update({
                    "rollout/ep_info_rew_mean": 0.0,
                })

            if self.int_rew_coef > 0:
                log_data.update({
                    "rollout/int_rew_mean": np.mean(self.ppo_rollout_buffer.intrinsic_rewards),
                    "rollout/int_rew_std": np.std(self.ppo_rollout_buffer.intrinsic_rewards),
                    "rollout/pos_int_rew_mean": np.maximum(self.ppo_rollout_buffer.intrinsic_rewards, 0.0).mean(),
                    "rollout/neg_int_rew_mean": np.minimum(self.ppo_rollout_buffer.intrinsic_rewards, 0.0).mean(),
                })

            if self.adv_norm > 0:
                log_data.update({
                    "rollout/adv_mean": np.mean(self.ppo_rollout_buffer.adv_mean),
                    "rollout/adv_std": np.std(self.ppo_rollout_buffer.adv_std),
                })

            # Update with other stats
            log_data.update(self.rollout_stats.to_dict())

            # Logging with wandb
            if self.use_wandb:
                wandb.log(log_data)
            # Logging with local logger
            if self.local_logger is not None:
                self.local_logger.write(log_data, log_type='rollout')

        # Log visualized data
        if self.enable_plotting and self.use_wandb:
            if self.iteration > 0 and self.iteration % self.plot_interval == 0:
                # Plot visit counts
                _, ax = plt.subplots()
                im = ax.imshow(
                    self.global_visit_counts,
                    cmap=self.plot_colormap,
                    interpolation='nearest')
                ax.figure.colorbar(im, ax=ax)
                plt.title(f'visit counts (iters:{self.iteration}, steps:{self.num_timesteps})')
                wandb.log({"visit counts": plt})
                plt.close()

                # Plot reward map
                _, ax = plt.subplots()
                im = ax.imshow(
                    self.global_reward_map_sums / (self.global_reward_map_nums + 1e-8),
                    cmap=self.plot_colormap,
                    interpolation='nearest')
                ax.figure.colorbar(im, ax=ax)
                plt.title(f'reward map (iters:{self.iteration}, steps:{self.num_timesteps})')
                wandb.log({"reward map": plt})
                plt.close()

                # Plot sum reward map
                _, ax = plt.subplots()
                im = ax.imshow(
                    self.global_reward_map_sums,
                    cmap=self.plot_colormap,
                    interpolation='nearest')
                ax.figure.colorbar(im, ax=ax)
                plt.title(f'sum reward map (iters:{self.iteration}, steps:{self.num_timesteps})')
                wandb.log({"sum reward map": plt})
                plt.close()

                # Plot max reward map
                _, ax = plt.subplots()
                im = ax.imshow(
                    self.global_reward_map_maxs,
                    cmap=self.plot_colormap,
                    interpolation='nearest')
                ax.figure.colorbar(im, ax=ax)
                plt.title(f'max reward map (iters:{self.iteration}, steps:{self.num_timesteps})')
                wandb.log({"max reward map": plt})
                plt.close()

                # Plot value map
                _, ax = plt.subplots()
                im = ax.imshow(
                    self.global_value_map_sums / (self.global_value_map_nums + 1e-8),
                    cmap=self.plot_colormap,
                    interpolation='nearest')
                ax.figure.colorbar(im, ax=ax)
                plt.title(f'value map (iters:{self.iteration}, steps:{self.num_timesteps})')
                wandb.log({"value map": plt})
                plt.close()

                # Clear counts
                self.global_visit_counts = np.zeros([self.width, self.height], dtype=np.int32)
                self.global_reward_map_maxs = np.zeros([self.width, self.height], dtype=np.float64)
                self.global_reward_map_sums = np.zeros([self.width, self.height], dtype=np.float64)
                self.global_reward_map_nums = np.zeros([self.width, self.height], dtype=np.int32)
                self.global_value_map_sums = np.zeros([self.width, self.height], dtype=np.float64)
                self.global_value_map_nums = np.zeros([self.width, self.height], dtype=np.int32)

    def get_expl_ratio(self):
        M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
        seq_observation = self.ppo_rollout_buffer.seq_obs_ratio[self.ppo_rollout_buffer.pos, :, :, :] # self.seq_obs:[32, 512, 5, 384], bert_input:[16384, 5, 384]
        seq_next_observation = self.ppo_rollout_buffer.seq_next_obs_ratio[self.ppo_rollout_buffer.pos, :, :, :] # self.seq_obs:[32, 512, 5, 384], bert_input:[16384, 5, 384]
        M, N, T, D = self.ppo_rollout_buffer.seq_act_ratio.shape
        seq_next_act = self.ppo_rollout_buffer.seq_act_ratio[self.ppo_rollout_buffer.pos, :, :, :] # self.seq_obs:[32, 512, 5, 23], bert_input:[16384, 5, 23]
        bert_loss, summed_attn, self.attn_all = self.bert_ratio(seq_observation.detach(), seq_next_act.detach().float(), seq_next_observation.detach(), keep_batch=True)
        # optimize BERT
        self.bert_ratio_opt.zero_grad(set_to_none=True)
        (bert_loss.mean()).backward()
        self.bert_ratio_opt.step()
        return summed_attn

    def create_intrinsic_rewards(self, new_obs, actions, dones):
        if self.int_rew_source == ModelType.NoModel:
            intrinsic_rewards = np.zeros([self.n_envs], dtype=float)
            model_mems = None
            return intrinsic_rewards, model_mems

        # Prepare input tensors for IR generation
        with th.no_grad():
            curr_obs_tensor = obs_as_tensor(self._last_obs, self.device)
            next_obs_tensor = obs_as_tensor(new_obs, self.device)
            curr_act_tensor = th.as_tensor(actions, dtype=th.int64, device=self.device)
            done_tensor = th.as_tensor(dones, dtype=th.int64, device=self.device)

            if self.policy.use_model_rnn: # true
                last_model_mem_tensor = self._last_model_mems
                if self.int_rew_source in [ModelType.RND, ModelType.NGU, ModelType.NovelD]:
                    if self.policy.rnd_use_policy_emb:
                        last_model_mem_tensor = self._last_policy_mems
            else:
                last_model_mem_tensor = None

            if self.policy.use_status_predictor: # false
                key_status_tensor = th.as_tensor(self.curr_key_status, dtype=th.int64, device=self.device)
                door_status_tensor = th.as_tensor(self.curr_door_status, dtype=th.int64, device=self.device)
                target_dists_tensor = th.as_tensor(self.curr_target_dists, dtype=th.float32, device=self.device)
            else:
                key_status_tensor = None
                door_status_tensor = None
                target_dists_tensor = None

        if self.const_repalce_ratio != 0:
            bert_ratio = self.const_repalce_ratio
            self.use_my_ratio=False
        else:
            if self.use_my_ratio:
                bert_ratio = self.get_expl_ratio()

            else:
                bert_ratio = 1

        # DEIR / Plain discriminator model
        if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]: # true
            intrinsic_rewards, model_mems = self.policy.int_rew_model.get_intrinsic_rewards( # model_mems:[16, 1, 128]
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                obs_history=self.episodic_obs_emb_history,
                trj_history=self.episodic_trj_emb_history,
                plain_dsc=bool(self.int_rew_source == ModelType.PlainDiscriminator),
            )
            # Insert obs into the Discriminator's obs queue
            # Algorithm A2 in the Technical Appendix
            if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]: # true
                self.policy.int_rew_model.update_obs_queue(
                    iteration=self.iteration,
                    intrinsic_rewards=intrinsic_rewards,
                    ir_mean=self.ppo_rollout_buffer.int_rew_stats.mean,
                    new_obs=new_obs,
                    stats_logger=self.rollout_stats
                )
            if self.use_my_ratio:
                if self.sequencelevel:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                    bert_ratio = bert_ratio.view(M, N, T)
                    bert_ratio = bert_ratio[self.ppo_rollout_buffer.pos]
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                else:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                bert_ratio = bert_ratio.view(*intrinsic_rewards.shape).detach().cpu().numpy()
            else:
                bert_ratio = 1
            intrinsic_rewards = intrinsic_rewards * bert_ratio
        # Plain forward / inverse model
        elif self.int_rew_source in [ModelType.PlainForward, ModelType.PlainInverse]:
            intrinsic_rewards, model_mems = self.policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_act=curr_act_tensor,
                curr_dones=done_tensor,
                obs_history=self.episodic_obs_emb_history,
                key_status=key_status_tensor,
                door_status=door_status_tensor,
                target_dists=target_dists_tensor,
                stats_logger=self.rollout_stats
            )
        # ICM
        elif self.int_rew_source == ModelType.ICM:
            intrinsic_rewards, model_mems = self.policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_act=curr_act_tensor,
                curr_dones=done_tensor,
                stats_logger=self.rollout_stats
            )
            if self.use_my_ratio:
                if self.sequencelevel:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                    bert_ratio = bert_ratio.view(M, N, T)
                    bert_ratio = bert_ratio[self.ppo_rollout_buffer.pos]
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                else:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                bert_ratio = bert_ratio.view(*intrinsic_rewards.shape).detach().cpu().numpy()
            else:
                bert_ratio = 1
            intrinsic_rewards = intrinsic_rewards * bert_ratio
        # RND
        elif self.int_rew_source == ModelType.RND:
            intrinsic_rewards, model_mems = self.policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_dones=done_tensor,
                stats_logger=self.rollout_stats
            ) # intrinsic_rewards:(16)
            if self.use_my_ratio:
                if self.sequencelevel:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                    bert_ratio = bert_ratio.view(M, N, T)
                    bert_ratio = bert_ratio[self.ppo_rollout_buffer.pos]
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                else:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                bert_ratio = bert_ratio.view(*intrinsic_rewards.shape).detach().cpu().numpy()
            else:
                bert_ratio = 1
            intrinsic_rewards = intrinsic_rewards * bert_ratio

        # NGU
        elif self.int_rew_source == ModelType.NGU:
            intrinsic_rewards, model_mems = self.policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_act=curr_act_tensor,
                curr_dones=done_tensor,
                obs_history=self.episodic_obs_emb_history,
                stats_logger=self.rollout_stats
            )
            if self.use_my_ratio:
                if self.sequencelevel:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                    bert_ratio = bert_ratio.view(M, N, T)
                    bert_ratio = bert_ratio[self.ppo_rollout_buffer.pos]
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                else:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                bert_ratio = bert_ratio.view(*intrinsic_rewards.shape).detach().cpu().numpy()
            else:
                bert_ratio = 1
            intrinsic_rewards = intrinsic_rewards * bert_ratio
        # NovelD
        elif self.int_rew_source == ModelType.NovelD:
            intrinsic_rewards, model_mems = self.policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_dones=done_tensor,
                stats_logger=self.rollout_stats
            )
            if self.use_my_ratio:
                if self.sequencelevel:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                    bert_ratio = bert_ratio.view(M, N, T)
                    bert_ratio = bert_ratio[self.ppo_rollout_buffer.pos]
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                else:
                    M, N, T, D = self.ppo_rollout_buffer.seq_obs_ratio.shape
                bert_ratio = bert_ratio.view(*intrinsic_rewards.shape).detach().cpu().numpy()
            else:
                bert_ratio = 1
            intrinsic_rewards = intrinsic_rewards * bert_ratio

        # MIMEx
        elif self.int_rew_source == ModelType.MIMEx:
            bert_loss, mask, loss_reward = self.policy.int_rew_model.get_intrinsic_rewards(
                self.ppo_rollout_buffer.seq_obs_mimex[self.ppo_rollout_buffer.pos, :, :, :].detach()
            )
            if self.use_my_ratio:
                if self.sequencelevel:
                    if loss_reward.shape[1] == bert_ratio.shape[1]:
                        loss_reward = loss_reward * bert_ratio.detach()
                    elif loss_reward.shape[1] < bert_ratio.shape[1]:
                        loss_reward *= bert_ratio[:, 0:loss_reward.shape[1]].detach()
                    else:
                        loss_reward[:, 0:bert_ratio.shape[1]] *= bert_ratio.detach()
                    loss_reward = (loss_reward * mask).sum(dim=-1) / mask.sum(dim=-1) # Take the prediction error of the mask part as the reward value
                else:
                    loss_reward *= bert_ratio.detach() # 先将[1024, 5]与[1024, 5]相乘

            if self.use_my_ratio:
                intrinsic_rewards = loss_reward.detach()
            else:
                intrinsic_rewards = loss_reward
            if self.anneal_k_mimex:
                anneal_weight = 1. - (self.num_timesteps / self.total_timesteps)
            else:
                anneal_weight = 1.
            intrinsic_rewards = intrinsic_rewards.cpu().numpy() * self.k_expl_mimex * anneal_weight
            return intrinsic_rewards, None
        else:
            raise NotImplementedError
        if self.record_ratio:
            self.alpha_ratio.append(bert_ratio)
        return intrinsic_rewards, model_mems

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        ppo_rollout_buffer: PPORolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        ppo_rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        self.init_on_rollout_start()


        while n_steps < n_rollout_steps:

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device) # 当前观察obs_tensor
                actions, values, log_probs, policy_mems = \
                    self.policy.forward(obs_tensor, self._last_policy_mems)
                actions = actions.cpu().numpy()
            save_intervel = np.random.randint(low=1, high=20)
            if self.collect_data and n_steps % save_intervel == 0:
                self.save_observations(self._last_obs)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Log before a transition
            self.log_before_transition(values)

            # Transition
            # Clipped_actions are collected as actions and integrated into the action sequence.
            # The next observation is new_obs, and done decides whether to eliminate elements from the sequence
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            if isinstance(new_obs, Dict):
                new_obs = new_obs["rgb"]
            if self.env_render:
                env.render()

            with th.no_grad():
                # Compute value for the last timestep
                new_obs_tensor = obs_as_tensor(new_obs, self.device)
                if dones[0] == True:
                    equal = torch.equal(obs_tensor[0], new_obs_tensor[0])
                _, new_values, _, _ = self.policy.forward(new_obs_tensor, policy_mems)
                seq_obs_feats = self.policy.obs_feature(obs_tensor) # obs_tensor[16, 64]
                seq_next_obs_feats = self.policy.obs_feature(obs_as_tensor(new_obs, self.device)) # obs_as_tensor
                
            if self.int_rew_source == ModelType.MIMEx:
                self.seq_obs_buffer_mimex[:, 0:-1] = self.seq_obs_buffer_mimex[:, 1:]
                self.seq_obs_buffer_mimex[:, -1] = seq_obs_feats.detach()
                seq_obs_mimex = self.seq_obs_buffer_mimex
                new_ids = (dones > 0).nonzero()
                self.seq_obs_buffer_mimex[new_ids] *= 0

                self.seq_next_obs_buffer_mimex[:, 0:-1] = self.seq_next_obs_buffer_mimex[:, 1:] # self.seq_obs_buffer:[512, 5, 384]
                self.seq_next_obs_buffer_mimex[:, -1] = seq_next_obs_feats.detach() # seq_obs_feats:[512, 384]
                seq_next_obs_mimex = self.seq_next_obs_buffer_mimex
                new_ids = (dones > 0).nonzero()
                self.seq_next_obs_buffer_mimex[new_ids] *= 0

                self.seq_act_buffer_mimex[:, 0:-1] = self.seq_act_buffer_mimex[:, 1:] # self.seq_obs_buffer:[512, 5, 384]
                self.seq_act_buffer_mimex[:, -1] = torch.from_numpy(clipped_actions) # seq_obs_feats:[512, 384]
                seq_act_mimex = self.seq_act_buffer_mimex
                new_ids = (dones > 0).nonzero()
                self.seq_act_buffer_mimex[new_ids] *= 0
            else:
                seq_obs_mimex = None
                seq_next_obs_mimex = None
                seq_act_mimex = None

            if self.use_my_ratio:
                self.seq_obs_buffer_ratio[:, 0:-1] = self.seq_obs_buffer_ratio[:, 1:]
                self.seq_obs_buffer_ratio[:, -1] = seq_obs_feats.detach()
                seq_obs_ratio = self.seq_obs_buffer_ratio
                new_ids = (dones > 0).nonzero()
                self.seq_obs_buffer_ratio[new_ids] *= 0

                self.seq_next_obs_buffer_ratio[:, 0:-1] = self.seq_next_obs_buffer_ratio[:, 1:] # self.seq_obs_buffer:[512, 5, 384]
                self.seq_next_obs_buffer_ratio[:, -1] = seq_next_obs_feats.detach() # seq_obs_feats:[512, 384]
                seq_next_obs_ratio = self.seq_next_obs_buffer_ratio
                new_ids = (dones > 0).nonzero()
                self.seq_next_obs_buffer_ratio[new_ids] *= 0

                self.seq_act_buffer_ratio[:, 0:-1] = self.seq_act_buffer_ratio[:, 1:] # self.seq_obs_buffer:[512, 5, 384]
                action_tmp = actions
                action_tmp = np.expand_dims(action_tmp, axis=1)
                self.seq_act_buffer_ratio[:, -1] = torch.from_numpy(action_tmp) # seq_obs_feats:[512, 384]
                seq_act_ratio = self.seq_act_buffer_ratio
                new_ids = (dones > 0).nonzero()
                self.seq_act_buffer_ratio[new_ids] *= 0
            else:
                seq_obs_ratio = None
                seq_next_obs_ratio = None
                seq_act_ratio = None
            if self.use_my_ratio or self.int_rew_source == ModelType.MIMEx:
                self.ppo_rollout_buffer.add_mimex_ratio(seq_obs_mimex, seq_act_mimex, seq_next_obs_mimex, seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio)
            # IR Generation
            intrinsic_rewards, model_mems = \
                self.create_intrinsic_rewards(new_obs, actions, dones) 
                # intrinsic_rewards:16, model_mems:[16, 1, 128], intrinsic_rewards scale

            # Log after the transition and IR generation
            self.log_after_transition(rewards, intrinsic_rewards)

            # Clear episodic memories when an episode ends
            self.clear_on_episode_end(dones, policy_mems, model_mems)

            # Update global stats
            self.num_timesteps += self.n_envs
            self._update_info_buffer(infos)
            n_steps += 1

            # Add to PPO buffer
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)
            ppo_rollout_buffer.add(
                self._last_obs,
                new_obs,
                self._last_policy_mems,
                self._last_model_mems,
                actions,
                rewards,
                intrinsic_rewards,
                self._last_episode_starts,
                dones,
                values,
                log_probs,
                self.curr_key_status,
                self.curr_door_status,
                self.curr_target_dists
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            if policy_mems is not None:
                self._last_policy_mems = policy_mems.detach().clone()
            if model_mems is not None:
                self._last_model_mems = model_mems.detach().clone()
        # Clear this rollback data
        ppo_rollout_buffer.step = 0
        ppo_rollout_buffer.compute_intrinsic_rewards()
        ppo_rollout_buffer.compute_returns_and_advantage(new_values, dones)
        callback.on_rollout_end()
        return True

    def save_observations(self, observation):
        for index, obs in enumerate(observation):
            # Transpose the observation data into the form of (height, width, channel) to adapt to the image format of PIL
            obs_transposed = np.transpose(obs, (1, 2, 0))
            # Convert the data type to uint8. If it is already present, this step is not necessary
            obs_uint8 = obs_transposed.astype(np.uint8)
            # Create a PIL image object
            img = Image.fromarray(obs_uint8)
            # Save image
            if self.image_noise_scale > 0:
                file_name = self.parent_dir_of_parent + '/noise_' + f"{self.img_num}.png"
            else:
                file_name = self.parent_dir_of_parent + '/' + f"{self.img_num}.png"
            img.save(file_name)
            self.img_num += 1

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "CustomOnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPORollout":
        self.iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.total_timesteps = total_timesteps
        callback.on_training_start(locals(), globals())
        self.on_training_start()
        print('Collecting rollouts ...')

        while self.num_timesteps < total_timesteps:
            collect_start_time = time.time()
            self.policy.eval()
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.ppo_rollout_buffer,
                n_rollout_steps=self.ppo_rollout_buffer.buffer_size)
            self.policy.train()
            collect_end_time = time.time()

            # Uploading rollout infos
            self.iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            self.log_on_rollout_end(log_interval)

            # Train the agent's policy and the IR model
            if continue_training is False:
                break
            train_start_time = time.time()
            self.train()
            train_end_time = time.time()

            # Print to the console
            rews = [ep_info["r"] for ep_info in self.ep_info_buffer]
            rew_mean = 0.0 if len(rews) == 0 else np.mean(rews)
            print(f'run: {self.run_id:2d}  '
                f'iters: {self.iteration}  '
                f'frames: {self.num_timesteps}  '
                f'rew: {rew_mean:.6f}  '
                f'rollout: {collect_end_time - collect_start_time:.3f} sec  '
                f'train: {train_end_time - train_start_time:.3f} sec')

        callback.on_training_end()
        return self