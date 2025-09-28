import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Any
import math, random, os, copy

from humanoidGym.learning.algorithms.normalizer import EmpiricalNormalization
from humanoidGym.learning.utils.helper import get_activation, mlp_factory
from humanoidGym.learning.utils.module import PureVqvaeEMA, StateHistoryEncoder
from humanoidGym.learning.utils.helper import smooth_decay, smooth_decay_se


# ========================= Adapter（残差，末层零初始化） =========================
class ResidualAdapter(nn.Module):
    """逐层残差注入，末层零初始化 → 初始等效恒等映射"""
    def __init__(self, h_dim: int, e_dim: int = 16, bottleneck: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(h_dim + e_dim, bottleneck)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(bottleneck, h_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, e], dim=-1)
        delta = self.fc2(self.act(self.fc1(x)))
        return h + delta
# ============================================================================


class InferenceActor(nn.Module):
    def __init__(self, actor_module, norm_module):
        super().__init__()
        self.actor_module = actor_module
        self.norm_module = norm_module
    def forward(self, x):
        x_norm = self.norm_module(x)
        y = self.actor_module(x_norm)
        return y


class MlpVqvaeSoftmaxLongEstLayerNormFallPredictRegressionTeacherVQSoftmaxActor(nn.Module):
    """
    这个类里包含三类关键改动：
    1) === 注入：z_dy(16) 逐层 Adapter 注入；
    2) === 自动冻结：首次 forward 自动只训练 Vae_dy + adapters；
    3) === 兼容旧 ckpt：覆写 `_load_from_state_dict` 做旧→新键名迁移 + 新键补齐（无需 pre-hook）。
    """
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super().__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist

        # =============== 基座 Actor（拆层，便于逐层注入） ===============
        # actor_input = [short(16) + long(16) + vel(3) + contact(2) + g(3)
        #                + obs + latents(latent_dim)]
        in_dim = 16 + 16 + 3 + 2 + 3 + (num_prop - 9) + 3 + latent_dim

        self.fc1 = nn.Linear(in_dim, 512)
        self.act1 = nn.ELU()
        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ELU()
        self.fc3 = nn.Linear(256, 128)
        self.ln3  = nn.LayerNorm(128)
        self.act3 = nn.ELU()
        self.head = nn.Linear(128, num_actions)

        # === 注入器（z_dy:16）
        self.ad1 = ResidualAdapter(512, e_dim=16)
        self.ad2 = ResidualAdapter(256, e_dim=16)
        self.ad3 = ResidualAdapter(128, e_dim=16)

        # =============== 其他模块 ===============
        self.Vae     = PureVqvaeEMA(in_dim=187,        output_dim=187,        num_emb=128)
        self.Vae_dy  = PureVqvaeEMA(in_dim=(num_prop-9)*50, output_dim=num_prop-9, num_emb=128)  # 输出 z_dy（dim=16）

        self.long_encoder  = StateHistoryEncoder(input_size=num_prop-9, tsteps=num_hist-1, output_size=16)

        self.short_encoder = nn.Sequential(
            nn.Linear((num_prop-9)*5, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 16), nn.ELU()
        )

        self.estimator_backbone = nn.Sequential(
            nn.Linear((num_prop-9)*5, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU()
        )

        self.predict_latent_layer = nn.Sequential(
            nn.Linear(64+16, 32), nn.ELU(),
            nn.Linear(32, latent_dim)
        )
        self.predict_vel_layer = nn.Sequential(
            nn.Linear(64+16, 32), nn.ELU(),
            nn.Linear(32, 3)
        )
        self.predict_contact_layer = nn.Sequential(
            nn.Linear(64+16, 32), nn.ELU(),
            nn.Linear(32, 2)
        )
        self.predict_gravity_vec_layer = nn.Sequential(
            nn.Linear(64+16, 32), nn.ELU(),
            nn.Linear(32, 3)
        )
        self.predict_layer = nn.Sequential(
            nn.Linear(16, 256), nn.ELU(),
            nn.Linear(256, 128)
        )

        self.random = 1

        # === MOD: 自动冻结控制标志（第一次 forward 自动冻结）
        self._zdy_freeze_done = False

    # === MOD: 兼容旧 ckpt：覆写 _load_from_state_dict（不需要 pre-hook，版本通用）
    def _load_from_state_dict(self, state_dict: Dict[str, Any], prefix: str, local_metadata, strict: bool,
                              missing_keys, unexpected_keys, error_msgs):
        """
        这里在父类真正加载前，对 `state_dict` 做就地修改：
        1) 旧 ckpt 的 actor.sequential -> fc1/fc2/fc3/ln3/head；
        2) 旧 ckpt 无 adapters / Vae_dy 时，补齐为当前模型参数（零初始化/默认初始化）；
        3) 保证 strict=True 也能通过。
        """
        # 1) 旧 -> 新 的键名映射
        old = f"{prefix}actor"
        new = f"{prefix}"
        key_map = {
            f"{old}.0.weight": f"{new}fc1.weight",
            f"{old}.0.bias":   f"{new}fc1.bias",
            f"{old}.2.weight": f"{new}fc2.weight",
            f"{old}.2.bias":   f"{new}fc2.bias",
            f"{old}.4.weight": f"{new}fc3.weight",
            f"{old}.4.bias":   f"{new}fc3.bias",
            f"{old}.5.weight": f"{new}ln3.weight",
            f"{old}.5.bias":   f"{new}ln3.bias",
            f"{old}.7.weight": f"{new}head.weight",
            f"{old}.7.bias":   f"{new}head.bias",
        }
        for k_old, k_new in key_map.items():
            if k_old in state_dict and k_new not in state_dict:
                state_dict[k_new] = state_dict[k_old]

        # 2) 删除旧 actor.* 键，避免 strict=True 报 unexpected
        for k in list(state_dict.keys()):
            if k.startswith(f"{old}."):
                del state_dict[k]

        # 3) 为新增的 Adapter 参数补齐（旧 ckpt 不会有）
        adapter_param_names = [
            "ad1.fc1.weight","ad1.fc1.bias","ad1.fc2.weight","ad1.fc2.bias",
            "ad2.fc1.weight","ad2.fc1.bias","ad2.fc2.weight","ad2.fc2.bias",
            "ad3.fc1.weight","ad3.fc1.bias","ad3.fc2.weight","ad3.fc2.bias",
        ]
        named_params = dict(self.named_parameters())
        for name in adapter_param_names:
            full = f"{prefix}{name}"
            if full not in state_dict and name in named_params:
                state_dict[full] = named_params[name].data.clone()

        # 4) 旧 ckpt 没有 Vae_dy / 其它本模块参数：用当前模型参数补齐（保证 strict=True 通过）
        cur = self.state_dict()
        for k, v in cur.items():
            k_full = f"{prefix}{k}"
            if k_full not in state_dict:
                state_dict[k_full] = v.clone()

        # 5) 调用父类真正加载
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def set_random(self, random):
        self.random = random

    def reshape(self, obs_hist_flatten):
        # N*(T*O) -> N*T*O
        obs_hist = obs_hist_flatten.reshape(-1, self.num_hist, self.num_prop)
        return obs_hist

    def reshape_critic(self, critic_obs_hist_flatten):
        height = critic_obs_hist_flatten[:, :187]
        critic_obs_hist = critic_obs_hist_flatten[:, 187:].reshape(-1, 5, 50 + 19 + 6 + 1)
        return critic_obs_hist, height

    # === MOD: 自动冻结：只训练 Vae_dy + Adapters（无需改上层 optimizer）
    def _auto_freeze_for_zdy(self):
        if self._zdy_freeze_done:
            return
        # 1) 全部冻结
        for p in self.parameters():
            p.requires_grad = False
        # 2) 仅打开 Vae_dy + 三个 Adapter
        for p in self.Vae_dy.parameters():
            p.requires_grad = True
        for p in self.ad1.parameters():
            p.requires_grad = True
        for p in self.ad2.parameters():
            p.requires_grad = True
        for p in self.ad3.parameters():
            p.requires_grad = True
        # 3) 保险：冻结参数的梯度清零 hook
        def _zero_grad_hook(grad):
            return torch.zeros_like(grad)
        frozen_modules = [
            self.Vae, self.long_encoder, self.short_encoder,
            self.estimator_backbone, self.predict_latent_layer,
            self.predict_vel_layer, self.predict_contact_layer,
            self.predict_gravity_vec_layer,
            self.fc1, self.fc2, self.fc3, self.ln3, self.head
        ]

        self._zdy_freeze_done = True

    def forward(self, obs_hist_flatten):
        # 第一次前向自动冻结（上层无需改 optimizer）
        if not self._zdy_freeze_done:
            self._auto_freeze_for_zdy()

        obs_hist = self.reshape(obs_hist_flatten)
        b, l, _ = obs_hist.size()

        # === z_dy 计算与注入（维度=16；取当前帧作为注入向量）
        vae_input = obs_hist[:, :-1, 9:].reshape(b,-1)  # [B*T, num_prop-9]
        

        short_hist_flatten = obs_hist[:, -5:, 9:].reshape(b, -1)
        short_encode = self.short_encoder(short_hist_flatten)  # [B,16]
        long_encode  = self.long_encoder(obs_hist[:, 1:, 9:])  # [B,16]

        with torch.no_grad():
            recon_dy, quantize_dy, z_dy, onehot_encode_dy = self.Vae_dy(vae_input)  # z_dy: [B*T, 16]
            encode = torch.cat([self.estimator_backbone(short_hist_flatten), long_encode], dim=-1)
            latents            = self.predict_latent_layer(encode)
            predicted_vel      = self.predict_vel_layer(encode)
            predicted_contact  = self.predict_contact_layer(encode)
            predicted_grad_vec = self.predict_gravity_vec_layer(encode)

        # 保持原有拼接（包含 long_encode）
        actor_input = torch.cat([
            short_encode,                    # 16
            long_encode,                     # 16
            predicted_vel.detach(),          # 3
            predicted_contact.detach(),      # 2
            predicted_grad_vec.detach(),     # 3
            obs_hist[:, -1, 9:],             # (num_prop-9)
            obs_hist[:, -1, 3:6],            # 3
            latents.detach()                 # latent_dim
        ], dim=-1)

        # 基座层 + 逐层 Adapter 注入 z_dy
        h = self.act1(self.fc1(actor_input))
        h = self.ad1(h, z_dy)

        h = self.act2(self.fc2(h))
        h = self.ad2(h, z_dy)

        h = self.act3(self.ln3(self.fc3(h)))
        h = self.ad3(h, z_dy)

        mean = self.head(h)
        return mean

    def get_latent(self, obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b, _, _ = obs_hist.size()

        short_hist_flatten = obs_hist[:, -5:, 9:].reshape(b, -1)
        long_encode = self.long_encoder(obs_hist[:, 1:, 9:])
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(short_hist_flatten), long_encode], dim=-1)
            latents = self.predict_latent_layer(encode)
            pred_class = torch.argmax(self.predict_layer(latents), dim=-1)
            print(f'地形类别：{pred_class[0]}')
        return latents, pred_class

    def VaeLoss(self, obs_hist_flatten, critic_obs_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist, height = self.reshape_critic(critic_obs_flatten)
        b, l, _ = obs_hist.size()

        # 静态高度图 VAE（已冻结；为了节省反传，这里直接 detach）
        recon, quantize, z, onehot_encode = self.Vae(height)
        rec_loss = self.Vae.loss_fn(height, recon, quantize, z, onehot_encode).detach()

        # 动力学 VAE（参与训练）
        vae_input = obs_hist[:, :-1, 9:].reshape(b, -1)
        vae_target = obs_hist[:, -1, 9:]
        recon_dy, quantize_dy, z_dy, onehot_encode_dy = self.Vae_dy(vae_input)
        reco_loss_dy = self.Vae_dy.loss_fn(vae_target, recon_dy, quantize_dy, z_dy, onehot_encode_dy)

        # regression（保持原逻辑；长编码 no_grad，已冻结）
        with torch.no_grad():
            long_encode = self.long_encoder(obs_hist[:, :-1, 9:])
        encode = torch.cat([self.estimator_backbone(obs_hist[:, -6:-1, 9:].reshape(b, -1)), long_encode], dim=-1)
        predict_latent = self.predict_layer(self.predict_latent_layer(encode))
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_gra_vec = self.predict_gravity_vec_layer(encode)

        latent_loss  = F.cross_entropy(predict_latent, torch.argmax(onehot_encode, dim=-1)).detach()
        mseloss      = F.mse_loss(predict_vel, obs_hist[:, -2, :3].detach())
        contact_loss = F.mse_loss(predict_contact, critic_hist[:, -2, -2:].detach())
        gravity_loss = F.mse_loss(predict_gra_vec, critic_hist[:, -1, 6:9].detach())

        return rec_loss, mseloss, latent_loss, contact_loss, gravity_loss, reco_loss_dy

    # 手动冻结（可选）
    def freeze_for_zdy_injection(self):
        self._zdy_freeze_done = False
        self._auto_freeze_for_zdy()

    # 仅返回 z_dy 相关可训练参数（可选：若你愿意上层改 optimizer 的话）
    def zdy_trainable_parameters(self):
        params = list(self.Vae_dy.parameters())
        params += list(self.ad1.parameters()) + list(self.ad2.parameters()) + list(self.ad3.parameters())
        return params


class ActorCriticAdapt(nn.Module):
    is_recurrent = False
    def __init__(self,
                 num_prop,
                 num_critic_obs,
                 num_hist,
                 num_actions,
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super().__init__()

        self.kwargs = kwargs
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_obs_h1 = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs

        self.actor_teacher_backbone = MlpVqvaeSoftmaxLongEstLayerNormFallPredictRegressionTeacherVQSoftmaxActor(
            num_prop=num_prop,
            num_hist=num_hist,
            num_actions=num_actions,
            actor_dims=[512, 256, 128],
            activation=activation,
            latent_dim=16
        )

        critic_layers = mlp_factory(activation, self.num_critic_obs, 1, critic_hidden_dims, last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def get_std(self):
        return self.std

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.act_inference(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_hist, **kwargs):
        mean = self.actor_teacher_backbone(obs_hist)
        return mean

    def get_latents(self, obs_hist):
        latent, pred_class = self.actor_teacher_backbone.get_latent(obs_hist)
        return latent, pred_class

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def set_random(self, it):
        random = smooth_decay_se(it, 3000, 2000, 1, 0.2)
        self.actor_teacher_backbone.set_random(random)

    # 可选：对外暴露同名接口（手动冻结 / 仅取可训练参数）
    def freeze_for_zdy_injection(self):
        self.actor_teacher_backbone.freeze_for_zdy_injection()

    def zdy_trainable_parameters(self):
        return self.actor_teacher_backbone.zdy_trainable_parameters()
