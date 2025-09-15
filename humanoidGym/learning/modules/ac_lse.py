from __future__ import annotations

from typing import List
import torch.nn as nn

import torch
from ..utils import resolve_nn_activation

from ..utils.helper import mlp_factory, tcn_factory
from ..utils import unpad_trajectories
from ..utils.module import StateHistoryEncoder

from .ac_slr import ActorCriticMlpSlrDblEnc
from .ac_base import AcNet, ActorCriticRnn, Memory




class InferenceActor(nn.Module):
    def __init__(self,actor_module,norm_module):
        super().__init__()
        self.actor_module = actor_module
        self.norm_module = norm_module
    def forward(self,x):
        x_norm = self.norm_module(x)
        y = self.actor_module(x_norm)
        return y

class ActorCriticLse(ActorCriticMlpSlrDblEnc):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_hist: int,
        num_actions: int,
        num_hist_short: int = 5,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        mlp_encoder_dims: list[int] = [256, 128, 64],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist=num_hist,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            mlp_encoder_dims=mlp_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
            **kwargs,
        )
        self.num_hist_short = num_hist_short
        self.vel_estimator = nn.Sequential(*mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=self.num_obs_h1 * num_hist_short + latent_dims,
            out_dims=3,
            hidden_dims=vel_encoder_dims,
        ))
        self.encoder = nn.Sequential(*mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=self.num_obs_h1 * self.num_hist,
            out_dims=latent_dims,
            hidden_dims=mlp_encoder_dims,
        ))

        self.encoder_critic = nn.Sequential(*mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=76+187,
            out_dims=latent_dims,
            hidden_dims=mlp_encoder_dims,
        ))

        self.actor = AcNet(
            is_policy=True,
            num_out=num_actions,
            num_obs=self.num_obs_h1 + self.num_latents  + 4 + 4 + 32 +2,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        self.critic = AcNet(
            is_policy=False,
            num_out=1,  # Critic output is a single value
            num_obs=self.num_obs_h1 + self.num_latents + 3 + 2 +3 +2, # 3 is linvel ,2 is phase,3 is gravc, 2 is contact 
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        self.long_encoder = StateHistoryEncoder(input_size=self.num_obs_h1,
                                             tsteps = num_hist-1,
                                             output_size=16)
        
        self.short_encoder = nn.Sequential(nn.Linear(self.num_obs_h1*self.num_hist_short,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,16),
                                           nn.ELU())
        
        self.estimator_backbone = nn.Sequential(nn.Linear(self.num_obs_h1*self.num_hist_short,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU())
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,2))
        
        self.predict_gravity_vec_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        self.random = 1
        
    def set_random(self,random):
        self.random = random

   
    


    def post_init(self):
        super().post_init()
        print(f"[INFO]: Velocity Estimator: {self.vel_estimator}")

    def extract(self, observations):
        obs_hist = observations.reshape(-1,self.num_hist,self.num_obs_h1+6) # add 3 for linvel, 2 for phase, 1 for stand
        obs_hist_prop = torch.cat([obs_hist[..., :, 9:],obs_hist[..., :, 3:6]],dim=-1) 
        prop = torch.cat([obs_hist[..., -1, 9:],obs_hist[..., -1, 3:8]],dim=-1) 
        hist_short = obs_hist_prop[..., -self.num_hist_short :, :]
        hist = obs_hist_prop[...,:,:]
        return hist, prop, hist_short

    def extract_critic(self, observations):
        height = observations[:,:187]
        critic_obs_hist = observations[:,187:].reshape(-1,5,50 + 19 + 6 + 1)
        # prop = torch.cat([critic_obs_hist[..., -1,9:51],critic_obs_hist[..., -1,3:6]],dim=-1)  # [Batch, Time, Dim]
        prop = torch.cat([critic_obs_hist[..., -1,9:51],critic_obs_hist[..., -1,3:6],critic_obs_hist[...,-1,6:8]],dim=-1)  # 相对于prop 添加phase
        vel = critic_obs_hist[...,-1, 0:3]
        contact = critic_obs_hist[...,-1,-2:]
        gravity = critic_obs_hist[...,-1,12:15]
        full_obs = torch.cat([height,critic_obs_hist[...,-1,:]],dim=-1)
        return full_obs, prop, vel, contact, gravity

    def encode(self, obs_tuple, **kwargs):
        obs_hist, _, obs_hist_short = obs_tuple
        short_encode = self.short_encoder(obs_hist_short.view(*obs_hist_short.shape[:-2],-1))
        long_encode = self.long_encoder(obs_hist[...,:-1,:])
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(obs_hist_short.view(*obs_hist_short.shape[:-2],-1)),long_encode],dim=-1)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            predicted_grad_vec = self.predict_gravity_vec_layer(encode)
            z = self.encoder(obs_hist.view(*obs_hist.shape[:-2], -1))  # TODO: 是否用和其他预测器一样encode
       
        return z, predicted_vel,predicted_contact,predicted_grad_vec,short_encode,long_encode

    def encode_critic(self, obs_tuple, **kwargs):
        full_obs, _, vel,contact, gravity = obs_tuple
        z = self.encoder_critic(full_obs.view(full_obs.shape[0], -1))
        return z, vel,contact, gravity

    def act(self, observations, **kwargs):
        mean = self.act_inference(observations, **kwargs)
        self.update_distribution(mean)
    
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        obs_tuple = self.extract(observations)
        prop = obs_tuple[1]
        z, predicted_vel,predicted_contact,predicted_grad_vec,short_encode,long_encode = self.encode(obs_tuple)
        actor_obs = torch.cat([z.detach(), prop, predicted_vel.detach(),predicted_contact.detach(),predicted_grad_vec.detach(),short_encode,long_encode], dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_tuple = self.extract_critic(critic_observations)
        prop = obs_tuple[1]
        z, vel,contact, gravity = self.encode_critic(obs_tuple)
        critic_obs = torch.cat([z, prop.squeeze(), vel.squeeze(),contact.squeeze(), gravity.squeeze()], dim=-1)
        value = self.critic(critic_obs)
        return value