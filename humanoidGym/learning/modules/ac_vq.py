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
from humanoidGym.learning.utils.module import PureVqvaeEMA,StateHistoryEncoder


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
    



class MlpVqvaeLongEstLayerNormFallPredictRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeLongEstLayerNormFallPredictRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32 + 3,512),
                                   nn.ELU(),
                                   nn.Linear(512,256),
                                   nn.ELU(),
                                   nn.Linear(256,128),
                                   nn.LayerNorm(128),
                                   nn.ELU(),
                                   nn.Linear(128,num_actions))
       
        self.Vae = PureVqvaeEMA(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=128)
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                             tsteps = num_hist-1,
                                             output_size=16)
        
        self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,16),
                                           nn.ELU())
        
        self.estimator_backbone = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64+16,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
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

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist
    
    def reshape_critic(self,critic_obs_hist_flatten):
        height = critic_obs_hist_flatten[:,:187]
        critic_obs_hist = critic_obs_hist_flatten[:,187:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
        return critic_obs_hist,height

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(short_hist_flatten),long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            predicted_grad_vec = self.predict_gravity_vec_layer(encode)
            
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),predicted_grad_vec.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
    
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,quantize,z,onehot_encode = self.Vae(vae_input)
        loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        
        encode = torch.cat([self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1)),long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_gra_vec = self.predict_gravity_vec_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:].detach())
        gravity_loss = F.mse_loss(predict_gra_vec,critic_hist[:,-1,6:9].detach())

      
        return   loss , mseloss , latent_loss , contact_loss , gravity_loss
    

class MlpVqvaeSoftmaxLongEstLayerNormFallPredictRegressionTeacherVQSoftmaxActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeSoftmaxLongEstLayerNormFallPredictRegressionTeacherVQSoftmaxActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32 + 3,512),
                                   nn.ELU(),
                                   nn.Linear(512,256),
                                   nn.ELU(),
                                   nn.Linear(256,128),
                                   nn.LayerNorm(128),
                                   nn.ELU(),
                                   nn.Linear(128,num_actions))
       
        
        self.Vae = PureVqvaeEMA(in_dim=187,output_dim=187,num_emb=128)
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                             tsteps = num_hist-1,
                                             output_size=16)
        
        self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,16),
                                           nn.ELU())
        
        self.estimator_backbone = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64+16,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,2))
        
        self.predict_gravity_vec_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_layer = nn.Sequential(nn.Linear(16,256),
                                                   nn.ELU(),
                                                    nn.Linear(256,128))
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist
    
    def reshape_critic(self,critic_obs_hist_flatten):
        height = critic_obs_hist_flatten[:,:187]
        critic_obs_hist = critic_obs_hist_flatten[:,187:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
        return critic_obs_hist,height

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

         
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(short_hist_flatten),long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            # pred_class = torch.argmax(self.predict_layer(latents), dim=-1) 
            # # print(f'地形类别：{pred_class}')
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            predicted_grad_vec = self.predict_gravity_vec_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,predicted_vel.detach(),predicted_contact.detach(),predicted_grad_vec.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],torch.zeros_like(latents).detach()],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
    
        return mean
    
    def get_latent(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

         
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(short_hist_flatten),long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            pred_class = torch.argmax(self.predict_layer(latents), dim=-1) 
            # print(f'地形类别：{pred_class[0]}')
       
    
        return latents, pred_class

    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        #VAE update
       
        recon,quantize,z,onehot_encode = self.Vae(height)
        rec_loss = self.Vae.loss_fn(height,recon,quantize,z,onehot_encode)

       
        
        # regression
        with torch.no_grad():
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        
        encode = torch.cat([self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1)),long_encode],dim=-1)
        predict_latent = self.predict_layer(self.predict_latent_layer(encode))
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_gra_vec = self.predict_gravity_vec_layer(encode)
        
        latent_loss = F.cross_entropy(predict_latent,torch.argmax(onehot_encode,dim=-1))
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:].detach())
        gravity_loss = F.mse_loss(predict_gra_vec,critic_hist[:,-1,6:9].detach())

    
        return rec_loss, mseloss , latent_loss , contact_loss , gravity_loss


class ActorCriticVq(nn.Module):
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
        super(ActorCriticVq, self).__init__()

        self.kwargs = kwargs

        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_obs_h1 = num_prop 
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs

        
        self.actor_teacher_backbone = MlpVqvaeSoftmaxLongEstLayerNormFallPredictRegressionTeacherVQSoftmaxActor(num_prop=num_prop,#remove linear vel
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