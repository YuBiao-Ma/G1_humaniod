from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Union


from ..modules.ac_vq import ActorCriticVq as ActorCritic

from .ppo import PPO


class PPOVQ(PPO):
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        aux_loss_coef=[1.0],
        # RND parameters
        rnd_cfg: Union[dict, None] = None,
        # Symmetry parameters
        symmetry_cfg: Union[dict, None] = None,
        
        value_smoothness_coef=0.1,
        smoothness_upper_bound=1.0,
        smoothness_lower_bound=0.1,
    ):
        super().__init__(
            actor_critic=actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,           
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            aux_loss_coef=aux_loss_coef,
            # RND parameters
            rnd_cfg = rnd_cfg,
            # Symmetry parameters
            symmetry_cfg = symmetry_cfg,
            value_smoothness_coef=value_smoothness_coef,
            smoothness_upper_bound=smoothness_upper_bound,
            smoothness_lower_bound=smoothness_lower_bound,
            )

        self.num_envs = 4096
        self.aux_loss_coef = aux_loss_coef

    def _compute_auxiliary_loss(self, batch: dict) -> dict:
        """Compute any auxiliary loss. Override this in subclasses if needed."""
        assert (
            self.num_envs is not None
        ), "[ERROR]: Number of environments must be provided for negative sample indexing."
        vqvae_loss,vel_loss , latent_loss,contact_loss , gravity_loss = self.actor_critic.actor_teacher_backbone.VaeLoss(batch["obs"],batch["critic_obs"])
       
        return {
            "vqvae": vqvae_loss * self.aux_loss_coef[0],
            "vel_mse": vel_loss * self.aux_loss_coef[-1],
            "latent_mse": latent_loss * self.aux_loss_coef[-1],
            "contact_mse": contact_loss * self.aux_loss_coef[-1],
            "gravity_mse": gravity_loss * self.aux_loss_coef[-1],

        }

   