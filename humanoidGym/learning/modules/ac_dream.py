import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Normal
import torch.optim as optim
import math
import random

import os
import copy

from humanoidGym.learning.algorithms.normalizer import EmpiricalNormalization

from humanoidGym.learning.utils.helper import get_activation, mlp_factory
from humanoidGym.learning.utils.module import PureVqvaeEMA,StateHistoryEncoder,PureBetaVAE


from humanoidGym.learning.utils.helper import smooth_decay, smooth_decay_se




class InferenceActor(nn.Module):
    def __init__(self,actor_module,norm_module):
        super().__init__()
        self.actor_module = actor_module
        self.norm_module = norm_module
    def forward(self,x):
        x_norm = self.norm_module(x)
        y = self.actor_module(x_norm)
        return y
    


    
class MlpBVAERegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAERegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=68, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=(num_prop-9)*50,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                             tsteps = num_hist-1,
                                             output_size=32)
        
        self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,32),
                                           nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,2))
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        vae_input = obs_hist[:,:-1,9:].reshape(b,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        
        with torch.no_grad():
            short_encode = self.short_encoder(obs_hist[:,-5:,9:].reshape(b,-1))
            long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
            encode = torch.cat([short_encode,long_encode],dim=-1)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)


        actor_input = torch.cat([mu,predicted_contact.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,6:8]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:-1,9:].reshape(b,-1)
        recon_target = obs_hist[:,-1,9:]
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        recon_loss = self.Vae.loss_fn(recon_target,recon,mu,log_var) # remove linvel and command
     
        
        # regression
        with torch.no_grad():
            _,_, mu,_ = self.Vae(vae_input)
        
        short_encode = self.short_encoder(obs_hist[:,-6:-1,9:].reshape(b,-1))
        long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        encode = torch.cat([short_encode,long_encode],dim=-1)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
      
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-1,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_obs_flatten[:,-2:])

        # loss = recon_loss + mseloss  + contact_loss
        return recon_loss,mseloss,contact_loss
    



class ActorCriticDream(nn.Module):
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
        super(ActorCriticDream, self).__init__()

        self.kwargs = kwargs

        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_obs_h1 = num_prop 
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs

        
        self.actor_teacher_backbone = MlpBVAERegressionActor(num_prop=num_prop,#remove linear vel
                                num_hist=num_hist,
                                num_actions=num_actions,
                                actor_dims=[512,256,128],
                                activation=activation,
                                latent_dim=16)
        
        
        # Value function
        critic_layers = mlp_factory(activation,self.num_critic_obs,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
      
    @staticmethod
    # not used at the moment
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

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self,obs_hist, **kwargs):
        mean = self.actor_teacher_backbone(obs_hist)
        return mean
    
    def get_latents(self,obs_hist):
        latent,pred_class= self.actor_teacher_backbone.get_latent(obs_hist)
        return latent,pred_class
    
    def evaluate(self, critic_observations, **kwargs):
        # critic_observations = self.critic_normalize(critic_observations)
        value = self.critic(critic_observations)
        return value
    
   
    
    def set_random(self,it):
        random = smooth_decay_se(it,3000,2000,1,0.2)    
        self.actor_teacher_backbone.set_random(random)