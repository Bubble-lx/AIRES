import os
import gym
import copy
import numpy as np
from torch.nn import GRUCell
from typing import Dict, Any, List, Union
from src.algo.intrinsic_rewards.deir import DiscriminatorModel
from src.algo.intrinsic_rewards.icm import ICMModel
from src.algo.intrinsic_rewards.ngu import NGUModel
from src.algo.intrinsic_rewards.mimex import MIMExModel
from src.algo.intrinsic_rewards.noveld import NovelDModel
from src.algo.intrinsic_rewards.plain_forward import PlainForwardModel
from src.algo.intrinsic_rewards.plain_inverse import PlainInverseModel
from src.algo.intrinsic_rewards.rnd import RNDModel
from src.algo.common_models.gru_cell import CustomGRUCell
from src.algo.common_models.mlps import *
from src.utils.common_func import init_module_with_name
from src.utils.enum_types import ModelType, NormType
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

class Encoder_obs(nn.Module):
    def __init__(self):
        super(Encoder_obs, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Output: 16x4x4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: 32x2x2
        self.fc = nn.Linear(32 * 2 * 2, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class PPOModel(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        learning_rate: float = 3e-4,
        model_learning_rate: float = 3e-4,
        run_id: int = 0,
        n_envs: int = 0,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        policy_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        policy_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_features_dim: int = 128,
        latents_dim: int = 128,
        model_latents_dim: int = 128,
        policy_mlp_norm: NormType = NormType.BatchNorm,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        policy_gru_norm: NormType = NormType.NoNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        use_status_predictor: int = 0,
        gru_layers: int = 1,
        policy_mlp_layers: int = 1,
        total_timesteps: int = 0,
        n_steps: int = 0,
        int_rew_source: ModelType = ModelType.DEIR,
        icm_forward_loss_coef: float = 0.2,
        ngu_knn_k: int = 10,
        ngu_dst_momentum: float = -1,
        ngu_use_rnd: int = 0,
        rnd_err_norm: int = 0,
        rnd_err_momentum: float = -1,
        rnd_use_policy_emb: int = 0,
        dsc_obs_queue_len: int = 0,
        log_dsc_verbose: int = 0,
        game_name: str = '',
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
        self.run_id = run_id
        self.n_envs = n_envs
        self.latents_dim = latents_dim
        self.activation_fn = activation_fn
        self.max_grad_norm = max_grad_norm
        self.model_features_dim = model_features_dim
        self.model_latents_dim = model_latents_dim
        self.action_num = action_space.n
        self.learning_rate = learning_rate
        self.model_learning_rate = model_learning_rate
        self.int_rew_source = int_rew_source
        self.policy_mlp_norm = policy_mlp_norm
        self.model_mlp_norm = model_mlp_norm
        self.model_cnn_norm = model_cnn_norm
        self.policy_gru_norm = policy_gru_norm
        self.model_gru_norm = model_gru_norm
        self.model_mlp_layers = model_mlp_layers
        self.gru_layers = gru_layers
        self.use_status_predictor = use_status_predictor
        self.policy_mlp_layers = policy_mlp_layers
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.policy_gru_cell = GRUCell if self.policy_gru_norm == NormType.NoNorm else CustomGRUCell
        self.model_gru_cell = GRUCell if self.model_gru_norm == NormType.NoNorm else CustomGRUCell
        self.use_model_rnn = use_model_rnn
        self.icm_forward_loss_coef = icm_forward_loss_coef
        self.ngu_knn_k = ngu_knn_k
        self.ngu_dst_momentum = ngu_dst_momentum
        self.ngu_use_rnd = ngu_use_rnd
        self.rnd_err_norm = rnd_err_norm
        self.rnd_err_momentum = rnd_err_momentum
        self.rnd_use_policy_emb = rnd_use_policy_emb
        self.policy_features_extractor_class = policy_features_extractor_class
        self.policy_features_extractor_kwargs = policy_features_extractor_kwargs
        self.model_cnn_features_extractor_class = model_cnn_features_extractor_class
        self.model_cnn_features_extractor_kwargs = model_cnn_features_extractor_kwargs
        self.dsc_obs_queue_len = dsc_obs_queue_len
        self.log_dsc_verbose = log_dsc_verbose
        self.game_name = game_name

        self.use_self_encoder=use_self_encoder

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

        self.use_next_obs = use_next_obs
        self.const_repalce_ratio = const_repalce_ratio
        self.attn_speed_way = attn_speed_way
        self.use_mask_ratio=use_mask_ratio
        self.seq_mask_value=seq_mask_value
        self.feat_mask=feat_mask
        self.feat_mask_value=feat_mask_value

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

        if isinstance(observation_space, gym.spaces.Dict):
            observation_space = observation_space["rgb"]

        self.dim_policy_features = self.policy_features_extractor_kwargs['features_dim']
        self.dim_model_features = self.model_features_dim

        self.policy_mlp_common_kwargs = dict(
            inputs_dim = self.dim_policy_features,
            latents_dim = self.latents_dim,
            activation_fn = self.activation_fn,
            mlp_norm = self.policy_mlp_norm,
            mlp_layers = self.policy_mlp_layers,
        )
        self.policy_rnn_kwargs = dict(
            input_size=self.dim_policy_features,
            hidden_size=self.dim_policy_features,
        )
        if self.policy_gru_norm != NormType.NoNorm:
            self.policy_rnn_kwargs.update(dict(
                norm_type=self.policy_gru_norm,
            ))

        super(ActorCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            False,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            self.policy_features_extractor_class,
            self.policy_features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self._init_modules()
        self._init_optimizers()


        # Build Intrinsic Reward Models
        int_rew_model_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            activation_fn=self.activation_fn,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            max_grad_norm=self.max_grad_norm,
            model_learning_rate=self.model_learning_rate,
            model_cnn_features_extractor_class=self.model_cnn_features_extractor_class,
            model_cnn_features_extractor_kwargs=self.model_cnn_features_extractor_kwargs,
            model_features_dim=self.model_features_dim,
            model_latents_dim=self.model_latents_dim,
            model_mlp_norm=self.model_mlp_norm,
            model_cnn_norm=self.model_cnn_norm,
            model_gru_norm=self.model_gru_norm,
            use_model_rnn=self.use_model_rnn,
            model_mlp_layers=self.model_mlp_layers,
            gru_layers=self.gru_layers,
            use_status_predictor=self.use_status_predictor,
        )

        if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
            self.int_rew_model = DiscriminatorModel(
                **int_rew_model_kwargs,
                obs_rng=np.random.default_rng(seed=self.run_id + 131),
                dsc_obs_queue_len=self.dsc_obs_queue_len,
                log_dsc_verbose=self.log_dsc_verbose,
            )
        if self.int_rew_source == ModelType.ICM:
            self.int_rew_model = ICMModel(
                icm_forward_loss_coef=self.icm_forward_loss_coef,
                **int_rew_model_kwargs,
            )
        if self.int_rew_source == ModelType.PlainForward:
            self.int_rew_model = PlainForwardModel(
                **int_rew_model_kwargs,
            )
        if self.int_rew_source == ModelType.PlainInverse:
            self.int_rew_model = PlainInverseModel(
                **int_rew_model_kwargs,
            )
        if self.int_rew_source == ModelType.RND:
            self.int_rew_model = RNDModel(
                **int_rew_model_kwargs,
                rnd_err_norm=self.rnd_err_norm,
                rnd_err_momentum=self.rnd_err_momentum,
                rnd_use_policy_emb=self.rnd_use_policy_emb,
                policy_cnn=self.features_extractor,
                policy_rnns=self.policy_rnns,
            )
        if self.int_rew_source == ModelType.NGU:
            self.int_rew_model = NGUModel(
                **int_rew_model_kwargs,
                ngu_knn_k=self.ngu_knn_k,
                ngu_dst_momentum=self.ngu_dst_momentum,
                ngu_use_rnd=self.ngu_use_rnd,
                rnd_err_norm=self.rnd_err_norm,
                rnd_err_momentum=self.rnd_err_momentum,
                rnd_use_policy_emb=self.rnd_use_policy_emb,
                policy_cnn=self.features_extractor,
                policy_rnns=self.policy_rnns,
            )
        if self.int_rew_source == ModelType.NovelD:
            self.int_rew_model = NovelDModel(
                **int_rew_model_kwargs,
                rnd_err_norm=self.rnd_err_norm,
                rnd_err_momentum=self.rnd_err_momentum,
                rnd_use_policy_emb=self.rnd_use_policy_emb,
                policy_cnn=self.features_extractor,
                policy_rnns=self.policy_rnns,
            )
        if self.int_rew_source == ModelType.MIMEx:
            self.int_rew_model = MIMExModel(
                **int_rew_model_kwargs,
                rnd_err_norm=self.rnd_err_norm,
                rnd_err_momentum=self.rnd_err_momentum,
                rnd_use_policy_emb=self.rnd_use_policy_emb,
                policy_cnn=self.features_extractor,
                policy_rnns=self.policy_rnns,
                use_my_ratio = self.use_my_ratio,
                attn_selet_way = self.attn_selet_way,
                sequencelevel = self.sequencelevel,
                ratio_s2aanda2s = self.ratio_s2aanda2s,
                attn_ratio_weight = self.attn_ratio_weight,

                seq_expl_len_ratio = self.seq_expl_len_ratio,
                decoder_depth_ratio = self.decoder_depth_ratio,
                norm_loss_ratio = self.norm_loss_ratio,
                use_cls_ratio = self.use_cls_ratio,
                decoder_num_heads_ratio = self.decoder_num_heads_ratio,
                bert_lr_ratio = self.bert_lr_ratio,
                # encoder_num_heads_ratio = self.encoder_num_heads_ratio,
                # encoder_depth_ratio = self.encoder_depth_ratio,

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
        if self.use_self_encoder:
            device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
            self.encoder_obs = Encoder_obs().to(device)
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)
            parent_dir_of_parent = os.path.dirname(current_dir)
            encoder_file_path = parent_dir_of_parent + "/pretrained/" + self.game_name + "_encoder.pth"
            assert os.path.exists(encoder_file_path), f"File does not exist: {encoder_file_path}"
            self.encoder_obs.load_state_dict(th.load(encoder_file_path))
        else:
            self.only_features_extractor = copy.deepcopy(self.features_extractor)


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PolicyValueOutputHeads(
            **self.policy_mlp_common_kwargs
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        # Build RNNs
        self.policy_rnns = []
        for l in range(self.gru_layers):
            name = f'policy_rnn_layer_{l}'
            setattr(self, name, self.policy_gru_cell(**self.policy_rnn_kwargs))
            self.policy_rnns.append(getattr(self, name))

    def _init_modules(self) -> None:
        nn.init.zeros_(self.action_net.weight)
        nn.init.zeros_(self.action_net.bias)
        nn.init.zeros_(self.value_net.weight)
        nn.init.zeros_(self.value_net.bias)

        module_names = {
            self.features_extractor: 'features_extractor',
            self.mlp_extractor: 'mlp_extractor',
        }

        for l in range(self.gru_layers):
            name = f'policy_rnn_layer_{l}'
            module = getattr(self, name)
            module_names.update({module: name})

        for module, name in module_names.items():
            init_module_with_name(name, module)

    def _init_optimizers(self) -> None:
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.learning_rate,
            **self.optimizer_kwargs
        )

    def _get_rnn_embeddings(self, hiddens: Optional[Tensor], inputs: Tensor, modules: List[nn.Module]):
        outputs = []
        for i, module in enumerate(modules):
            hidden_i = th.squeeze(hiddens[:, i, :])
            output_i = module(inputs, hidden_i)
            inputs = output_i
            outputs.append(output_i)
        outputs = th.stack(outputs, dim=1)
        return outputs

    def _get_latent(self, obs: Tensor, mem: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        curr_features = self.features_extractor(obs) # curr_features:[16, 64]
        memories = self._get_rnn_embeddings(mem, curr_features, self.policy_rnns)
        features = th.squeeze(memories[:, -1, :])
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_sde = latent_pi
        return latent_pi, latent_vf, latent_sde, memories

    def forward(self, obs: Tensor, mem: Tensor, deterministic: bool = False) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        latent_pi, latent_vf, latent_sde, memories = self._get_latent(obs, mem)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, memories

    def obs_feature(self, obs):
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images) # obs is [16 3 7 7]
        if self.use_self_encoder:
            curr_features = self.encoder_obs(obs) # my_encoder
        else:
            curr_features = self.only_features_extractor(obs) # obs is [16, 64] random_encoder_no change
        return curr_features

    def evaluate_policy(self, obs: Tensor, act: Tensor, mem: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        latent_pi, latent_vf, latent_sde, memories = self._get_latent(obs, mem)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(act)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), memories
