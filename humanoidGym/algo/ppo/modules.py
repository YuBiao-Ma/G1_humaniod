import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import random

import os
import copy

from humanoidGym.algo.ppo.normalizer import EmpiricalNormalization

def get_activation(act_name):
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")
    
def mlp_factory(activation, input_dims, out_dims, hidden_dims,last_act=False):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    if last_act:
        layers.append(activation)

    return layers

def spectral_mlp_factory(activation, input_dims, out_dims, hidden_dims,last_act=False):
    layers = []
    layers.append(nn.utils.spectral_norm(nn.Linear(input_dims, hidden_dims[0])))
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dims[l], hidden_dims[l + 1])))
        layers.append(activation)

    if out_dims:
        layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dims[-1], out_dims)))
    if last_act:
        layers.append(activation)

    return layers


def mlp_batchnorm_factory(activation, input_dims, out_dims, hidden_dims,last_act=False,bias=True):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0],bias=bias))
    layers.append(nn.BatchNorm1d(hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1],bias=bias))
        layers.append(nn.BatchNorm1d(hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims,bias=bias))
    if last_act:
        layers.append(activation)

    return layers
    
class BetaVAE(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 beta: int = 0.1) -> None:
        
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_vel = nn.Linear(encoder_hidden_dims[-1],3)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),

                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)

        self.kl_weight = beta

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        vel = self.fc_vel(result)

        return [mu,log_var,vel]
    
    def get_latent(self,input):
        mu,log_var,vel = self.encode(input)
        #z = self.reparameterize(mu, log_var)
        return mu,vel

    def decode(self,z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        
        mu,log_var,vel = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z),z, mu, log_var, vel]
    
    def loss_fn(self,y, y_hat, mean, logvar):
     
        # recons_loss = 0.5*F.mse_loss(y_hat,y,reduction="none").sum(dim=-1)
        # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), -1)
        # loss = (recons_loss + self.beta * kl_loss).mean(dim=0)

        recons_loss = F.mse_loss(y_hat,y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), -1))
        loss = recons_loss + self.beta * kl_loss

        return loss

# class MixedMlp(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         latent_size,
#         hidden_size,
#         num_actions,
#         num_experts,
#     ):
#         super().__init__()

#         input_size = latent_size + input_size
#         inter_size = hidden_size + latent_size
#         output_size = num_actions

#         self.mlp_layers = [
#             (
#                 nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
#                 nn.Parameter(torch.empty(num_experts, hidden_size)),
#                 F.elu,
#             ),
#             (
#                 nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
#                 nn.Parameter(torch.empty(num_experts, hidden_size)),
#                 F.elu,
#             ),
#             (
#                 nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
#                 nn.Parameter(torch.empty(num_experts, output_size)),
#                 None,
#             ),
#         ]

#         for index, (weight, bias, _) in enumerate(self.mlp_layers):
#             index = str(index)
#             torch.nn.init.kaiming_uniform_(weight)
#             bias.data.fill_(0.01)
#             self.register_parameter("w" + index, weight)
#             self.register_parameter("b" + index, bias)

#         # Gating network
#         gate_hsize = 64
#         self.gate = nn.Sequential(
#             nn.Linear(input_size, gate_hsize),
#             nn.ELU(),
#             nn.Linear(gate_hsize, gate_hsize),
#             nn.ELU(),
#             nn.Linear(gate_hsize, num_experts),
#         )

#     def forward(self, z, c):
#         #c = self.c_norm(c)

#         coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
#         layer_out = c
#         for (weight, bias, activation) in self.mlp_layers:
#             flat_weight = weight.flatten(start_dim=1, end_dim=2)
#             mixed_weight = torch.matmul(coefficients, flat_weight).view(
#                 coefficients.shape[0], *weight.shape[1:3]
#             )

#             input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
#             mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
#             out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
#             layer_out = activation(out) if activation is not None else out

#         return layer_out
    
class MixedMlp(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        hidden_size,
        num_actions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + input_size
        inter_size = hidden_size + latent_size
        output_size = num_actions

        self.mlp_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.mlp_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.LayerNorm(gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.LayerNorm(gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts)
        )

    def forward(self, z, c):
        #c = self.c_norm(c)

        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        
        layer_out = c
        for (weight, bias, activation) in self.mlp_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out

class MlpBVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAEActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = BetaVAE(in_dim=5*(num_prop-8),beta=0.1,output_dim=num_prop-8) #remove baselin and command
        # self.obs_normalizer = EmpiricalNormalization(shape=num_prop)
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,5:,8:].reshape(b,-1)) #remove linvel and command
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        # obs_hist = obs_hist[:,5:,:].view(b,-1)
        recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,4:-1,8:].reshape(b,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,-1,8:],recon,mu,log_var) # remove linvel and command
        # mseloss = F.mse_loss(predicted_vel,priv)
        mseloss = F.mse_loss(predicted_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = loss + mseloss
        return loss
    
class MlpRnnBVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpRnnBVAEActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = BetaRnnVAE(in_dim=num_prop-8,beta=0.1,output_dim=num_prop-8) #remove baselin and command
        # self.obs_normalizer = EmpiricalNormalization(shape=num_prop)
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,:,8:]) #remove linvel and command
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        # obs_hist = obs_hist[:,5:,:].view(b,-1)
        recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,:-1,8:]) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,-1,8:],recon,mu,log_var) # remove linvel and command
        mseloss = F.mse_loss(predicted_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = loss + mseloss
        return loss

class MlpBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 mlp_encoder_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.mlp_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=(num_prop-6)*5,
                                 out_dims=None,
                                 hidden_dims=mlp_encoder_dims))
        
        self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                   #nn.BatchNorm1d(32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.mlp_encoder(obs_hist[:,5:,6:].reshape(b,-1)) #remove linvel and command
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.mlp_encoder(obs_hist[:,5:,6:].reshape(b,-1))
        z2 = self.mlp_encoder(obs_hist[:,4:-1,6:].reshape(b,-1))

        z1_l = self.latent_layer(z1)
        z1_v = self.vel_layer(z1)

        z2_l = self.latent_layer(z2)

        z1_l = self.projector(z1_l) 
        z2_l = self.projector(z2_l)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(z1_v,0.1*obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss
        
        return loss
    
# class MlpBarlowTwinsRegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  num_actions,
#                  actor_dims,
#                  mlp_encoder_dims,
#                  latent_dim,
#                  activation) -> None:
#         super(MlpBarlowTwinsRegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
        
        
#         # barlow related
#         self.mlp_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
#                                  input_dims=(num_prop-8)*5,
#                                  out_dims=None,
#                                  hidden_dims=mlp_encoder_dims))
#         self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
#                                           nn.BatchNorm1d(32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
#         # future related 
#         self.predict_mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=(num_prop-8)*5,
#                                  out_dims=None,
#                                  hidden_dims=mlp_encoder_dims))
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
#         self.predict_vel_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))

#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop,
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
        
#         self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
#                                  input_dims=latent_dim,
#                                  out_dims=64,
#                                  hidden_dims=[64],
#                                  bias=False))
        
#         self.bn = nn.BatchNorm1d(64,affine=False)

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             latent = self.predict_mlp_encoder(obs_hist[:,5:,8:].reshape(b,-1)) #remove linvel and command
#             z = self.predict_latent_layer(latent)
#             vel = self.predict_vel_layer(latent)
            
#         actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
        
#     def BarlowTwinsLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()

#         z1 = self.mlp_encoder(obs_hist[:,5:,8:].reshape(b,-1))
#         z2 = self.mlp_encoder(obs_hist[:,4:-1,8:].reshape(b,-1))

#         z1_l_ = self.latent_layer(z1)
#         z2_l_ = self.latent_layer(z2)

#         z1_l = self.projector(z1_l_) 
#         z2_l = self.projector(z2_l_)

#         c = self.bn(z1_l).T @ self.bn(z2_l)
#         c.div_(b)

#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag = off_diagonal(c).pow_(2).sum()
        
#         pred_z = self.predict_mlp_encoder(obs_hist[:,4:-1,8:].reshape(b,-1))
#         pred_z_l = self.predict_latent_layer(pred_z)
#         pred_vel = self.predict_vel_layer(pred_z)

#         latent_loss = F.mse_loss(pred_z_l,z1_l_)
#         priv_loss = F.mse_loss(pred_vel,0.1*obs_hist[:,-2,:3].detach())

#         loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
#         return loss

class MlpBarlowTwinsRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 mlp_encoder_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.mlp_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=(num_prop-6)*5,
                                 out_dims=None,
                                 hidden_dims=mlp_encoder_dims))
        self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=(num_prop-6)*5,
                                 out_dims=None,
                                 hidden_dims=mlp_encoder_dims))
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_mlp_encoder(obs_hist[:,5:,6:].reshape(b,-1)) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.mlp_encoder(obs_hist[:,5:,6:].reshape(b,-1))
        z2 = self.mlp_encoder(obs_hist[:,4:-1,6:].reshape(b,-1))

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_mlp_encoder(obs_hist[:,4:-1,6:].reshape(b,-1))
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MixedMlpBarlowTwinsRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 mlp_encoder_dims,
                 latent_dim,
                 activation) -> None:
        super(MixedMlpBarlowTwinsRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.mlp_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=(num_prop-8)*5,
                                 out_dims=None,
                                 hidden_dims=mlp_encoder_dims))
        self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=(num_prop-8)*5,
                                 out_dims=None,
                                 hidden_dims=mlp_encoder_dims))
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        # self.actor = nn.Sequential(*mlp_factory(activation=activation,
        #                          input_dims=latent_dim + num_prop,
        #                          out_dims=num_actions,
        #                          hidden_dims=actor_dims))
        
        self.actor = MixedMlp(input_size=num_prop-3,
                              latent_size=latent_dim+3,
                              hidden_size=128,
                              num_actions=num_actions,
                              num_experts=4)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_mlp_encoder(obs_hist[:,5:,8:].reshape(b,-1)) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        # actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        latents = torch.cat([z.detach(),vel.detach()],dim=-1)
        mean  = self.actor(latents,obs_hist[:,-1,3:])
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.mlp_encoder(obs_hist[:,5:,8:].reshape(b,-1))
        z2 = self.mlp_encoder(obs_hist[:,4:-1,8:].reshape(b,-1))

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_mlp_encoder(obs_hist[:,4:-1,8:].reshape(b,-1))
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
    
class MlpBarlowTwinsRnnRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsRnnRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.rnn_encoder = RnnStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=num_prop-8,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=128)
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_rnn_encoder = RnnStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=num_prop-8,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=128)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_rnn_encoder(obs_hist[:,:,8:]) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.rnn_encoder(obs_hist[:,1:,8:])
        z2 = self.rnn_encoder(obs_hist[:,:-1,8:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_rnn_encoder(obs_hist[:,:-1,8:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss

def off_diagonal(x):
    n,m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()
    
class MlpRnnBarlowTwinActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpRnnBarlowTwinActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.rnn_encoder = RnnStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=num_prop-6,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=128)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(128,32),
                                   #nn.BatchNorm1d(32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
       
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():            
            latent = self.rnn_encoder(obs_hist[:,:,6:])
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        obs_hist = obs_hist.detach()

        z1 = self.rnn_encoder(obs_hist[:,1:,6:])
        z2 = self.rnn_encoder(obs_hist[:,:-1,6:])

        z1_l = self.latent_layer(z1)
        z1_v = self.vel_layer(z1)

        z2_l = self.latent_layer(z2)

        z1_l = self.projector(z1_l) 
        z2_l = self.projector(z2_l)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(z1_v,0.1*obs_hist[:,-1,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss
        
        return loss
    
class MlpRnnFullBVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpRnnFullBVAEActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = BetaFullRnnVAE(in_dim=num_prop-8,beta=0.1,output_dim=num_prop-8) #remove baselin and command
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,:,8:]) #remove linvel and command
        actor_input = torch.cat([latents[:,-1,:].detach(),predicted_vel[:,-1,:].detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        # obs_hist = obs_hist[:,5:,:].view(b,-1)
        recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,:-1,8:]) # remove linvel and command
        mseloss = F.mse_loss(predicted_vel,0.1*obs_hist[:,:-1,:3].detach())
        loss = self.Vae.loss_fn(obs_hist[:,1:,8:],recon,mu,log_var) # remove linvel and command

        loss = loss + mseloss
        return loss

class Quantizer(nn.Module):
    def __init__(self,embedding_dim,num_embeddings):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )
        self.linear_proj = nn.Linear(embedding_dim,int(embedding_dim/2))

    def forward(self, z: torch.Tensor):
        # z_norm = F.normalize(self.linear_proj(z))
        # emb_norm = F.normalize(self.linear_proj(self.embeddings.weight))

        z_norm = F.normalize(z)
        emb_norm = F.normalize(self.embeddings.weight)

        distances = (
            (z_norm ** 2).sum(dim=-1, keepdim=True)
            + (emb_norm**2).sum(dim=-1)
            - 2 * z_norm @ emb_norm.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings).type_as(z_norm)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ emb_norm

        return quantized
    
class VQVAE_vel(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 num_emb = 32) -> None:
        
        super(VQVAE_vel, self).__init__()

        self.latent_dim = latent_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_vel = nn.Sequential(nn.Linear(encoder_hidden_dims[-1], 3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=num_emb)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        z = self.quantizer(z)
        return z,vel

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        z = F.normalize(z)
        vel = self.fc_vel(latent)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss

class QuantizerEMA(nn.Module):
    def __init__(self,embedding_dim,num_embeddings):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = 0.99

        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        embeddings.data.normal_()

        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer(
            "ema_embed", torch.zeros(self.num_embeddings, self.embedding_dim)
        )

        self.register_buffer("embeddings", embeddings)

        self.linear_proj = nn.Linear(embedding_dim,int(embedding_dim/2))

    # def update_codebook(self,z,one_hot_encoding):
    #     n_i = torch.sum(one_hot_encoding, dim=0)

    #     self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

    #     dw = one_hot_encoding.T @ z.reshape(-1, self.embedding_dim)

    #     ema_embed = self.ema_embed * self.decay + dw * (1 - self.decay)

    #     n = torch.sum(self.cluster_size)

    #     self.cluster_size = (
    #         (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
    #     )

    #     self.embeddings.data.copy_(ema_embed / self.cluster_size.unsqueeze(-1))
    #     self.ema_embed.data.copy_(ema_embed)

    # def forward(self, z: torch.Tensor):

    #     z_ = self.linear_proj(z)
    #     emb_ = self.linear_proj(self.embeddings)

    #     # z_norm = F.normalize(z_, dim=-1) 
    #     # emb_norm = F.normalize(emb_,dim=-1)

    #     distances = (
    #         (z_ ** 2).sum(dim=-1, keepdim=True)
    #         + (emb_**2).sum(dim=-1)
    #         - 2 * z_ @ emb_.T
    #     )

    #     closest = distances.argmin(-1).unsqueeze(-1)

    #     one_hot_encoding = (
    #         F.one_hot(closest, num_classes=self.num_embeddings)
    #         .type(torch.float)
    #         .squeeze(1)
    #     )

    #     # quantization
    #     quantized = one_hot_encoding @ self.embeddings


    #     return quantized,one_hot_encoding

    def update_codebook(self, z, one_hot_encoding):
        n_i = torch.sum(one_hot_encoding, dim=0)
        self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

        dw = one_hot_encoding.T @ z.reshape(-1, self.embedding_dim)
        ema_embed = self.ema_embed * self.decay + dw * (1 - self.decay)

        n = torch.sum(self.cluster_size)
        self.cluster_size = ((self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n)

        self.embeddings.data.copy_(ema_embed / self.cluster_size.unsqueeze(-1))
        self.ema_embed.data.copy_(ema_embed)

        # ---- 追加：复活长期未使用的 code ----
        dead = self.cluster_size < 1e-3
        if dead.any():
            num_dead = int(dead.sum().item())
            # 随机从当前 batch 的 z 中采样新向量作为复活初始化
            rand_idx = torch.randint(0, z.shape[0], (num_dead,), device=z.device)
            new_vecs = z[rand_idx].detach()
            self.embeddings.data[dead] = new_vecs
            self.ema_embed.data[dead]  = new_vecs
            self.cluster_size.data[dead] = 1.0

    def forward(self, z: torch.Tensor):
        # 用原空间的余弦距离做最近邻选择，避免与返回向量空间不一致
        with torch.no_grad():
            z_n   = F.normalize(z, dim=-1)
            emb_n = F.normalize(self.embeddings, dim=-1)
            # 余弦距离：2 - 2*cos
            distances = 2 - 2 * (z_n @ emb_n.T)              # (B, K)
            closest = distances.argmin(-1)                   # (B,)
            one_hot_encoding = F.one_hot(
                closest, num_classes=self.num_embeddings
            ).type(z.dtype)                                  # (B, K)

        # 查表得到量化向量（用未归一化的原始 codebook）
        quantized = one_hot_encoding @ self.embeddings       # (B, D)
        return quantized, one_hot_encoding

    
class VQVAE_Rnn(nn.Module):

    def __init__(self,
                 in_dim= 45,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE_Rnn, self).__init__()

        self.latent_dim = latent_dim
        
        self.encoder = RnnStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=in_dim,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=128)

        self.fc_mu = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, latent_dim))

        self.fc_vel = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, 3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        z = self.quantizer(z)
        return z,vel

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        z = F.normalize(z)
        vel = self.fc_vel(latent)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss
    
class VQVAE_EMA_Rnn(nn.Module):

    def __init__(self,
                 in_dim= 45,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 num_emb=32) -> None:
        
        super(VQVAE_EMA_Rnn, self).__init__()

        self.latent_dim = latent_dim
        
        self.encoder = RnnStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=in_dim,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=128)

        self.fc_mu = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, latent_dim))

        self.fc_vel = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, 3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=num_emb)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        #z,_ = self.quantizer(z)
        return z,vel

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        z = F.normalize(z)
        vel = self.fc_vel(latent)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize,onehot_encode = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel,onehot_encode
    
    def loss_fn(self,y, y_hat,quantized,z,onehot_encode):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        # embedding_loss = F.mse_loss(
        #     quantized,
        #     z.detach(),
        #     reduction="sum",
        # )
        self.quantizer.update_codebook(z,onehot_encode)

        vq_loss = 0.25*commitment_loss #+ embedding_loss

        return recon_loss + vq_loss

# class MlpVQVAEActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpVQVAEActor,self).__init__()
#         self.num_hist = num_hist
#         self.num_prop = num_prop

#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop,
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.Vae = VQVAE_vel(in_dim=5*(num_prop-8),output_dim=(num_prop-8))
        
#     def normalize_reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten.detach()
#         #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
#         return obs_hist

        
#     def forward(self,obs_hist_flatten):
#         obs_hist = self.normalize_reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             latents,predicted_vel = self.Vae.get_latent(obs_hist[:,5:,8:].reshape(b,-1))

#         actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.normalize_reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()

#         recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,4:-1,8:].reshape(b,-1))
#         loss = self.Vae.loss_fn(obs_hist[:,-1,8:],recon,quantize,z) 
#         mseloss = F.mse_loss(predicted_vel,0.1*obs_hist[:,-2,:3].detach())
#         loss = loss + mseloss

#         return loss

# class MlpVQVAERnnActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpVQVAERnnActor,self).__init__()
#         self.num_hist = num_hist
#         self.num_prop = num_prop

#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop,
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.Vae = VQVAE_EMA_Rnn(in_dim=num_prop-8,output_dim=num_prop-8)
        
#     def normalize_reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten.detach()
#         #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
#         return obs_hist

        
#     def forward(self,obs_hist_flatten):
#         obs_hist = self.normalize_reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             latents,predicted_vel = self.Vae.get_latent(obs_hist[:,:,8:])

#         actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.normalize_reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()

#         recon,quantize,z,predicted_vel,onehot_encode = self.Vae(obs_hist[:,:-1,8:])
#         loss = self.Vae.loss_fn(obs_hist[:,-1,8:],recon,quantize,z,onehot_encode) 
#         mseloss = F.mse_loss(predicted_vel,0.1*obs_hist[:,-2,:3].detach())
#         loss = loss + mseloss

#         return loss
    
class RnnStateHistoryEncoder(nn.Module):
    def __init__(self,activation_fn, input_size,mlp_output_size, encoder_dims,hidden_size):
        super(RnnStateHistoryEncoder,self).__init__()
        self.activation_fn = activation_fn
        self.encoder_dims = encoder_dims
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(*mlp_factory(activation=activation_fn,
                                   input_dims=input_size,
                                   hidden_dims=encoder_dims,
                                   out_dims=mlp_output_size))
        
        self.rnn = nn.GRU(input_size=mlp_output_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           num_layers = 1)
        
    def forward(self,obs):
        h_0 = torch.zeros(1,obs.size(0),self.hidden_size,device=obs.device).requires_grad_()
        obs = self.encoder(obs)
        out, h_n = self.rnn(obs,h_0)
        return out[:,-1,:]
    
class RnnFullStateHistoryEncoder(nn.Module):
    def __init__(self,activation_fn, input_size,mlp_output_size, encoder_dims,hidden_size):
        super(RnnFullStateHistoryEncoder,self).__init__()
        self.activation_fn = activation_fn
        self.encoder_dims = encoder_dims
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(*mlp_factory(activation=activation_fn,
                                   input_dims=input_size,
                                   hidden_dims=encoder_dims,
                                   out_dims=mlp_output_size))
        
        self.rnn = nn.GRU(input_size=mlp_output_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           num_layers = 1)
        
    def forward(self,obs):
        h_0 = torch.zeros(1,obs.size(0),self.hidden_size,device=obs.device).requires_grad_()
        obs = self.encoder(obs)
        out, h_n = self.rnn(obs,h_0)
        return out
    
    # def inference(self,obs,hidden):
    #     # hidden : 1,obs.size(0),self.hidden_size
    #     obs = self.encoder(obs)
    #     _,h_n = self.rnn(obs,hidden)
    #     return h_n
    
class BetaRnnVAE(nn.Module):

    def __init__(self,
                 in_dim= 42,
                 latent_dim = 16,
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 beta: int = 0.1) -> None:
        
        super(BetaRnnVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = RnnStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=in_dim,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=128)

        self.fc_mu = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, latent_dim))
        self.fc_var = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, latent_dim))
        self.fc_vel = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, 3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)

        self.kl_weight = beta

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        vel = self.fc_vel(result)

        return [mu,log_var,vel]
    
    def get_latent(self,input):
        mu,log_var,vel = self.encode(input)
        return mu,vel

    def decode(self,z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        
        mu,log_var,vel = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z),z, mu, log_var, vel]
    
    def loss_fn(self,y, y_hat, mean, logvar):
     
        recons_loss = F.mse_loss(y_hat,y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), -1))
        loss = recons_loss + self.beta * kl_loss

        return loss
    
class BetaFullRnnVAE(nn.Module):

    def __init__(self,
                 in_dim= 42,
                 latent_dim = 16,
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 beta: int = 0.1) -> None:
        
        super(BetaFullRnnVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = RnnFullStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=in_dim,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=128)

        self.fc_mu = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, latent_dim))
        self.fc_var = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, latent_dim))
        self.fc_vel = nn.Sequential(nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64, 3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)

        self.kl_weight = beta

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        vel = self.fc_vel(result)

        return [mu,log_var,vel]
    
    def get_latent(self,input):
        mu,log_var,vel = self.encode(input)
        return mu,vel

    def decode(self,z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        
        mu,log_var,vel = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z),z, mu, log_var, vel]
    
    def loss_fn(self,y, y_hat, mean, logvar):
     
        recons_loss = F.mse_loss(y_hat,y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), -1))
        loss = recons_loss + self.beta * kl_loss

        return loss
    
class CnnHistoryEncoder(nn.Module):
    def __init__(self, input_size, tsteps):
        # self.device = device
        super(CnnHistoryEncoder, self).__init__()
        self.tsteps = tsteps

        channel_size = 16
        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), 
                nn.ELU(),
                nn.Linear(3 * channel_size,3 * channel_size)
                )

        self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2),
                nn.ELU(),
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1),
                nn.ELU(),
                nn.Flatten())

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        return output

# class CnnHistoryEncoder(nn.Module):
#     def __init__(self, input_size, tsteps):
#         # self.device = device
#         super(CnnHistoryEncoder, self).__init__()
#         self.tsteps = 66#tsteps

#         channel_size = 16
#         self.encoder = nn.Sequential(
#                 nn.Linear(input_size, 3 * channel_size), 
#                 nn.ELU(),
#                 nn.Linear(3 * channel_size,3 * channel_size)
#                 )

#         self.conv_layers = nn.Sequential(
#                 nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 3),
#                 nn.ELU(),
#                 nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2),
#                 nn.ELU(),
#                 nn.Flatten())

#     def forward(self, obs):
#         # nd * T * n_proprio
#         nd = obs.shape[0]
#         T = self.tsteps
#         projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
#         output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
#         return output

"""
long hist cnn
"""
class LongHistCnn(nn.Module):
    def __init__(self,num_obs,in_channels,output_size,filter_size,stride_size,kernel_size):
        super(LongHistCnn,self).__init__()
        
        self.num_obs = num_obs
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        
        long_history_layers = []
        cnn_output_dim = self.num_obs
        for out_channels, kernel_size, stride_size in zip(self.filter_size, self.kernel_size, self.stride_size):
            long_history_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride_size))
            long_history_layers.append(nn.ReLU())
            cnn_output_dim = (cnn_output_dim - kernel_size + stride_size) // stride_size
            in_channels = out_channels
        cnn_output_dim *= out_channels
        long_history_layers.append(nn.Flatten())
        long_history_layers.append(nn.Linear(cnn_output_dim, 128))
        long_history_layers.append(nn.ELU())
        long_history_layers.append(nn.Linear(128, output_size))
        self.long_history = nn.Sequential(*long_history_layers)
    
    def forward(self, obs):
        return self.long_history(obs.view(-1, self.in_channels, self.num_obs))
        
#MlpBarlowTwinsCnnRegressionNoPhaseActor
class MlpBarlowTwinsCnnRegressionNoPhaseActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionNoPhaseActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        actor_list = mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 128,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        latent = self.predict_cnn_encoder(obs_hist[:,1:,6:]) #remove linvel and command
        
        with torch.no_grad():
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
        actor_input = torch.cat([latent,z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,6:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,6:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,6:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpSimSiamActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(MlpSimSiamActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop + latent_dim,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

        # past encoder
        self.encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(128,32),
                                       nn.ELU(),
                                       nn.Linear(32,3))
        # future encoder
        self.future_encoder = nn.Sequential(nn.Linear(num_prop-8,128),
                                    nn.BatchNorm1d(128),
                                    nn.ELU(),
                                    nn.Linear(128,64),
                                    nn.BatchNorm1d(64),
                                    nn.ELU(),
                                    nn.Linear(64,latent_dim))
        
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(16, 8),
                                        nn.BatchNorm1d(8),
                                        nn.ELU(inplace=True), # hidden layer
                                        nn.Linear(8, 16)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1)
        
        self.phase_gate = PhaseGate(num_prop + latent_dim - 2)
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist
        
    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.encoder(obs_hist[:,1:,8:])
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)

        cmd = obs_hist[:,-1,3:6]
        no_phase_part = torch.cat([z.detach(),vel.detach(),cmd,obs_hist[:,-1,8:].detach()],dim=-1)
        phase_part = obs_hist[:,-1,6:8]
        
        phase_gated = self.phase_gate(no_phase_part.detach(),phase_part)
        # cmd = 
        actor_input = torch.cat([no_phase_part.detach(),phase_gated],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def SimSiamLoss(self,obs_hist_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z2 = self.future_encoder(obs_hist[:,-1,8:])
        z1_ = self.encoder(obs_hist[:,:-1,8:])

        z1_vel = self.vel_layer(z1_)
        z1 = self.latent_layer(z1_)
        
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        # z1 = z1.detach()
        # z2 = z2.detach()

        priv_loss = F.mse_loss(z1_vel,obs_hist[:,-2,:3].detach())
        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5 
        loss = loss + priv_loss 
        return loss
    
class PhaseGate(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # 用其他状态特征生成门控信号（0-1之间）
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, 64),  # 排除phase后的状态维度
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()  # 输出门控值
        )
    
    def forward(self, state, phase):
        gate = self.gate_net(state)

        return gate * phase  # 当gate≈0时，phase被忽略

class PhaseGen(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # 用其他状态特征生成门控信号（0-1之间）
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, 64),  # 排除phase后的状态维度
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64,2),
            nn.Tanh()  # 输出门控值
        )
    
    def forward(self, state):
        gate = self.gate_net(state)
        return gate  # 当gate≈0时，phase被忽略
    
class DualBranchActor(nn.Module):
    def __init__(self, actor_phase, actor_nophase, input_size):
        super().__init__()
        self.branch_with_phase = actor_phase
        self.branch_without_phase = actor_nophase
        self.gate_net = nn.Sequential(
            nn.Linear(input_size, 64),  # 排除phase后的状态维度
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()  # 输出门控值
        )
    
    def forward(self, state_nophase, state_phase):
        action1 = self.branch_with_phase(state_phase)
        action2 = self.branch_without_phase(state_nophase)
        weight = self.gate_net(state_nophase)
        
        return weight * action1 + (1 - weight) * action2
    
class MlpSimSiamSingleStepActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(MlpSimSiamSingleStepActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop + latent_dim,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.actor_nophase = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop + latent_dim,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.encoder = nn.Sequential(nn.Linear(num_prop-8,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,latent_dim))
        # future encoder
        self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(128,32),
                                       nn.ELU(),
                                       nn.Linear(32,3))
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(16, 8),
                                        nn.BatchNorm1d(8),
                                        nn.ELU(inplace=True), # hidden layer
                                        nn.Linear(8, 16)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1)
    
        self.random = 1.0
            
    def set_random(self,random):
        self.random = random
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist
        
    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_encoder(obs_hist[:,1:,8:])
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)

        if random.random() < self.random:
            cmd = obs_hist[:,-1,3:6]
            phase = obs_hist[:,-1,6:8]     
            input_phase = torch.cat([z.detach(),vel.detach(),cmd,phase,obs_hist[:,-1,8:].detach()],dim=-1)
            mean = self.actor(input_phase)
        else:
            cmd = obs_hist[:,-1,3:6]
            phase = obs_hist[:,-1,6:8]*0
            input_no_phase = torch.cat([z.detach(),vel.detach(),cmd,phase,obs_hist[:,-1,8:].detach()],dim=-1)
            mean = self.actor_nophase(input_no_phase)
            
        return mean
    
    def SimSiamLoss(self,obs_hist_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,d = obs_hist.size()

        z2 = self.encoder(obs_hist[:,1:,8:].reshape(b*(l-1),-1))
        z1 = self.encoder(obs_hist[:,:-1,8:].reshape(b*(l-1),-1))
        
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        
        with torch.no_grad():
            future_latent = self.encoder(obs_hist[:,-1,8:])
            
        z = self.predict_encoder(obs_hist[:,:-1, 8:])
        z_pred = self.latent_layer(z)
        vel_pred = self.vel_layer(z)
        
        priv_loss = F.mse_loss(vel_pred,obs_hist[:,-2,:3].detach())
        latent_loss = F.mse_loss(z_pred,future_latent.detach())
        
        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5 
        loss = loss + priv_loss + latent_loss 
        
        # mimic phase and no phase
        with torch.no_grad():
            cmd = obs_hist[:,-2,3:6]
            phase = obs_hist[:,-2,6:8]
            phase_zero = torch.zeros_like(phase)     
            input_phase = torch.cat([z_pred.detach(),vel_pred.detach(),cmd,phase,obs_hist[:,-2,8:].detach()],dim=-1)
            input_no_phase = torch.cat([z_pred.detach(),vel_pred.detach(),cmd,phase_zero,obs_hist[:,-2,8:].detach()],dim=-1)
        target_action = self.actor(input_phase)
        action = self.actor_nophase(input_no_phase)
        loss += self.random*F.mse_loss(action,target_action)
        
        return loss
    
class MlpSimSiamSingleStepNoPhaseActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(MlpSimSiamSingleStepNoPhaseActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop + 128 + latent_dim,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.encoder = nn.Sequential(nn.Linear(num_prop-6,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,latent_dim))
        # future encoder
        self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(128,32),
                                       nn.ELU(),
                                       nn.Linear(32,3))
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(16, 8),
                                        nn.BatchNorm1d(8),
                                        nn.ELU(inplace=True), # hidden layer
                                        nn.Linear(8, 16)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1)
    
        self.random = 1.0
            
    def set_random(self,random):
        self.random = random
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist
        
    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        # with torch.no_grad():
        latent = self.predict_encoder(obs_hist[:,1:,6:])
        z = self.latent_layer(latent)
        vel = self.vel_layer(latent)

        input = torch.cat([latent, z.detach(),vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
        mean = self.actor(input)
            
        return mean
    
    def SimSiamLoss(self,obs_hist_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,d = obs_hist.size()

        z2 = self.encoder(obs_hist[:,1:,6:].reshape(b*(l-1),-1))
        z1 = self.encoder(obs_hist[:,:-1,6:].reshape(b*(l-1),-1))
        
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        
        with torch.no_grad():
            future_latent = self.encoder(obs_hist[:,-1,6:])
            
        z = self.predict_encoder(obs_hist[:,:-1, 6:])
        z_pred = self.latent_layer(z)
        vel_pred = self.vel_layer(z)
        
        priv_loss = F.mse_loss(vel_pred,obs_hist[:,-2,:3].detach())
        latent_loss = F.mse_loss(z_pred,future_latent.detach())
        
        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5 
        loss = loss + priv_loss + latent_loss 
        
        return loss
    
class MlpSimSiamSingleStepHeightActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(MlpSimSiamSingleStepHeightActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop + latent_dim*2 - 187,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

        self.encoder = nn.Sequential(nn.Linear(num_prop-8-187,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,latent_dim))
        # future encoder
        self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-8-187,
                                             tsteps = 20)        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.height_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.height_encoder = nn.Sequential(nn.Linear(187,128),
                                          nn.BatchNorm1d(128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,latent_dim))
        
        self.vel_layer = nn.Sequential(nn.Linear(128,32),
                                       nn.ELU(),
                                       nn.Linear(32,3))
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(16, 8),
                                        nn.BatchNorm1d(8),
                                        nn.ELU(inplace=True), # hidden layer
                                        nn.Linear(8, 16)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1)
        
        self.random = 1.0
            
    def set_random(self,random):
        self.random = random
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist
        
    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_encoder(obs_hist[:,1:,195:])
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
        #if random.random() < self.random:
        # heights_latent = self.height_encoder(obs_hist[:,-1,:187])
        # else:
        with torch.no_grad():
            heights_latent = self.height_latent_layer(latent)
            heights_latent = heights_latent.detach()
        
        phase = obs_hist[:,-1,193:195] #* self.random
        cmd = obs_hist[:,-1,190:193]
        input_phase = torch.cat([z.detach(),vel.detach(),heights_latent,cmd,phase,obs_hist[:,-1,195:].detach()],dim=-1)

        mean  = self.actor(input_phase)
        return mean
    
    def SimSiamLoss(self,obs_hist_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,d = obs_hist.size()

        z2 = self.encoder(obs_hist[:,1:,195:].reshape(b*(l-1),-1))
        z1 = self.encoder(obs_hist[:,:-1,195:].reshape(b*(l-1),-1))
        
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        
        with torch.no_grad():
            future_latent = self.encoder(obs_hist[:,-1,195:])
            height_latent = self.height_encoder(obs_hist[:,-2,:187])
            
        z = self.predict_encoder(obs_hist[:,:-1, 195:])
        z_pred = self.latent_layer(z)
        vel_pred = self.vel_layer(z)
        height_pred = self.height_latent_layer(z)
        
        priv_loss = F.mse_loss(vel_pred,obs_hist[:,-2,187:190].detach())
        latent_loss = F.mse_loss(z_pred,future_latent)
        height_loss = F.mse_loss(height_pred,height_latent)
        
        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5 
        loss = loss + priv_loss + latent_loss + height_loss
    
        return loss
    
class MlpHistoryHeightNoPhaseActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(MlpHistoryHeightNoPhaseActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop
        
        self.height_size = 187
        self.prop_start = 3
        self.prop_nocmd_start = 6
        self.history_size = 128
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop - self.height_size + latent_dim + self.history_size,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-self.prop_nocmd_start-self.height_size,
                                             tsteps = 20)
                
        self.height_latent_layer = nn.Sequential(nn.Linear(self.history_size,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.height_encoder = nn.Sequential(nn.Linear(self.height_size,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU(),
                                          nn.Linear(64,latent_dim))
        
        self.vel_layer = nn.Sequential(nn.Linear(self.history_size,32),
                                       nn.ELU(),
                                       nn.Linear(32,3))
        self.random = 1.0
            
    def set_random(self,random):
        self.random = random
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist
        
    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        latent = self.predict_encoder(obs_hist[:,1:,self.height_size+self.prop_nocmd_start:])
        with torch.no_grad():
            vel = self.vel_layer(latent)
            
        # if random.random() < self.random:
        #     heights_latent = self.height_encoder(obs_hist[:,-1,:self.height_size])
        # else:
        with torch.no_grad():
            heights_latent = self.height_latent_layer(latent)
            heights_latent = heights_latent.detach()
        
        input = torch.cat([latent,vel.detach(),heights_latent,obs_hist[:,-1,self.height_size+3:].detach()],dim=-1)

        mean  = self.actor(input)
        return mean
    
    def HeightLatentLoss(self,obs_hist_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,d = obs_hist.size()
        
        # from history from 0 to last time frame
        z = self.predict_encoder(obs_hist[:,:-1, self.height_size+self.prop_nocmd_start:])
        vel_pred = self.vel_layer(z)
        height_pred = self.height_latent_layer(z)
        
        height_latent = self.height_encoder(obs_hist[:,-2,:self.height_size])
        
        # speed
        priv_loss = F.mse_loss(vel_pred,obs_hist[:,-2,self.height_size:self.height_size+3].detach())
        # terrian
        latent_loss = F.mse_loss(height_pred,height_latent)
        
        loss = priv_loss + latent_loss 
    
        return loss
    
class MlpBarlowTwinsCnnRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-9,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-9,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
        # # # short history
        # self.hist_mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
        #                          input_dims=(num_prop-8)*4,
        #                          out_dims=None,
        #                          hidden_dims=[128,32]))

    def set_random(self,random):
        pass
    
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,9:]) #remove linvel and command

            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        # short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,9:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,9:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,9:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpBarlowTwinsCnnRegressionDirectPastActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionDirectPastActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop - 3)*5 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,8:]) #remove linvel and command

            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        # short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-5:,3:].reshape(b, -1)],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,8:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,8:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,8:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpBarlowTwinsCnnRegressionDirectPastNoPhaseActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionDirectPastNoPhaseActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop - 3)*5 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,6:]) #remove linvel and command

            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        # short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-5:,3:].reshape(b, -1)],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,6:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,6:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,6:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
    
class MlpBarlowTwinsCnnSingleActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnSingleActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.next_encoder = nn.Sequential(nn.Linear(num_prop-8,128),
                                    nn.BatchNorm1d(128),
                                    nn.ELU(),
                                    nn.Linear(128,64),
                                    nn.BatchNorm1d(64),
                                    nn.ELU(),
                                    nn.Linear(64,latent_dim))
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.cnn_encoder(obs_hist[:,1:,8:]) #remove linvel and command

            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
            
        # short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1_l_ = self.next_encoder(obs_hist[:,-1,8:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,8:])

        z2_l_ = self.latent_layer(z2)
        pred_vel = self.vel_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss
        
        return loss
    
    
class MlpBarlowTwinsCnnSingleNoPhaseActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnSingleNoPhaseActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)
        
        self.next_encoder = nn.Sequential(nn.Linear(num_prop-8,128),
                                    nn.BatchNorm1d(128),
                                    nn.ELU(),
                                    nn.Linear(128,64),
                                    nn.BatchNorm1d(64),
                                    nn.ELU(),
                                    nn.Linear(64,latent_dim))
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.cnn_encoder(obs_hist[:,1:,6:]) #remove linvel and command

            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
            
        # short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1_l_ = self.next_encoder(obs_hist[:,-1,6:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,6:])

        z2_l_ = self.latent_layer(z2)
        pred_vel = self.vel_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss
        
        return loss
    
    
class MlpBarlowTwinsCnnDeltaRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnDeltaRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        # self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
        #                                      tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
        # # # short history
        # self.hist_mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
        #                          input_dims=(num_prop-8)*4,
        #                          out_dims=None,
        #                          hidden_dims=[128,32]))

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.cnn_encoder(obs_hist[:,1:,8:]) #remove linvel and command
            # current_latent = self.cnn_encoder(obs_hist[:,1:,8:])
            # z_current = self.predict_latent_layer(current_latent)
            z = self.latent_layer(latent)
            delta = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        # short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),delta.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,8:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,8:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z_l = self.predict_latent_layer(z2.detach()) + z2_l_.detach()
        pred_vel = self.predict_vel_layer(z2.detach())

        latent_loss = F.mse_loss(pred_z_l,z1_l_.detach())
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
    
class MlpBarlowTwinsCnnRegressionCurrentActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionCurrentActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        actor_list = mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        # actor_list.append(nn.Tanh())
        
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,8:]) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
            current_latent = self.cnn_encoder(obs_hist[:,1:,8:])
            z_current = self.latent_layer(current_latent)

        actor_input = torch.cat([z.detach(),z_current.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,8:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,8:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,8:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpBarlowTwinsCnnRegressionShortHistActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionShortHistActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        actor_list = mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        # actor_list.append(nn.Tanh())
        
        self.actor = nn.Sequential(*actor_list)
        #self.actor = add_sn(self.actor)
        print(self.actor)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
        # # # short history
        self.hist_mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=(num_prop-8)*4,
                                 out_dims=None,
                                 hidden_dims=[128,64,16]))

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,8:]) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),short_hist,obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,8:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,8:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,8:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpBarlowTwinsCnnRegressionShortHistActorNophase(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionShortHistActorNophase,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        # actor_list = spectral_mlp_factory(activation=activation,
        #                          input_dims=latent_dim + num_prop + 16,
        #                          out_dims=num_actions,
        #                          hidden_dims=actor_dims)
        actor_list = mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        # actor_list.append(nn.Tanh())
        
        self.actor = nn.Sequential(*actor_list)
        #self.actor = add_sn(self.actor)
        print(self.actor)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
        # # # short history
        self.hist_mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=(num_prop-6)*4,
                                 out_dims=None,
                                 hidden_dims=[128,64,16]))

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,6:]) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,6:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),short_hist,obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,6:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,6:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,6:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
    
# class MlpBarlowTwinsCnnRegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  num_actions,
#                  actor_dims,
#                  latent_dim,
#                  activation) -> None:
#         super(MlpBarlowTwinsCnnRegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
        
        
#         # barlow related
#         self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
#                                              tsteps = 20)
        
#         self.latent_layer = nn.Sequential(nn.Linear(128,32),
#                                           nn.BatchNorm1d(32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
#         # future related 
#         self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-6,
#                                              tsteps = 20)
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
#         self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))

#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop,
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
        
#         self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
#                                  input_dims=latent_dim,
#                                  out_dims=64,
#                                  hidden_dims=[64],
#                                  bias=False))
        
#         self.bn = nn.BatchNorm1d(64,affine=False)

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             latent = self.predict_cnn_encoder(obs_hist[:,1:,6:]) #remove linvel and command
#             z = self.predict_latent_layer(latent)
#             vel = self.predict_vel_layer(latent)
            
#         actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
        
#     def BarlowTwinsLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()

#         z1 = self.cnn_encoder(obs_hist[:,1:,6:])
#         z2 = self.cnn_encoder(obs_hist[:,:-1,6:])

#         z1_l_ = self.latent_layer(z1)
#         z2_l_ = self.latent_layer(z2)

#         z1_l = self.projector(z1_l_) 
#         z2_l = self.projector(z2_l_)

#         c = self.bn(z1_l).T @ self.bn(z2_l)
#         c.div_(b)

#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag = off_diagonal(c).pow_(2).sum()
        
#         pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,6:])
#         pred_z_l = self.predict_latent_layer(pred_z)
#         pred_vel = self.predict_vel_layer(pred_z)

#         latent_loss = F.mse_loss(pred_z_l,z1_l_)
#         priv_loss = F.mse_loss(pred_vel,0.1*obs_hist[:,-2,:3].detach())

#         loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
#         return loss

"""
Transformer
"""
class StateCausalTransformer(nn.Module):
    def __init__(self, n_obs,n_embd,n_head,n_layer,dropout,block_size,bias=False) -> None:
        super().__init__()
        
        # obs embedding
        self.embedding = nn.Sequential(
            nn.Linear(n_obs,n_embd),
            nn.GELU(),
            nn.Linear(n_embd,n_embd)
        )
        # transformer 
        self.transformer = nn.ModuleDict(dict(
            wpe = PositionalEncoding(n_embd,dropout=0.0),
            h = nn.ModuleList([Block(n_embd,n_head,bias,dropout,block_size) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, obs_history):

        # get embedding 
        tok_emb = self.embedding(obs_history)

        x = self.transformer.wpe(tok_emb.permute(1,0,2))
        x = x.permute(1,0,2)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd,n_head,bias,dropout,block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class MLP(nn.Module):

    def __init__(self, n_embd,bias,dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 2 * n_embd, bias= bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(2 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, n_embd,n_head,bias,dropout,block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd,n_head,bias,dropout,block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd,bias,dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class MlpTransRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpTransRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        # transformer policy
        self.trans_backbone = StateCausalTransformer(n_obs=num_prop-3,
                                                     n_embd=64,
                                                     n_head=4,
                                                     n_layer=2,
                                                     dropout=0,
                                                     block_size=10,
                                                     bias=False)
        
        self.action_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, num_actions)
        )
        
        self.next_obs_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, num_prop)
        )

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        latent = self.trans_backbone(obs_hist[:,1:,3:])
        mean  = self.action_head(latent[:,-1,:])
        return mean
        
    def NextObsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        # get latent
        latent = self.trans_backbone(obs_hist[:,:-1,3:].detach())
        predicted_obs = self.next_obs_head(latent)
        
        reg_loss = F.mse_loss(predicted_obs,obs_hist[:,1:,:].detach())

        return reg_loss
    
class MlpBarlowTwinsCnnRegressionDirectPastTwoPhaseActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnRegressionDirectPastTwoPhaseActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = CnnHistoryEncoder(input_size=num_prop-8,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop - 3)*5 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        # no phase
        self.student_actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop - 5)*5 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,8:]) #remove linvel and command

            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        # short_hist = self.hist_mlp_encoder(obs_hist[:,-5:-1,8:].reshape(b,-1))
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-5:,3:].reshape(b, -1)],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,8:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,8:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,8:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss 
        
        return loss
    
class MlpBaselineActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBaselineActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        self.cnn_encoder = LongHistCnn(num_obs=num_prop-3,
                                       in_channels=num_hist,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.vel_encoder = nn.Sequential(nn.Linear((num_prop-3) * 5,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU(),
                                          nn.Linear(64,3))
    
        actor_list = mlp_factory(activation=activation,
                                 input_dims= (num_prop-3) * 5 + 64 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        short_hist = obs_hist[:,-5:,3:].reshape(b,-1)
        long_hist = self.cnn_encoder(obs_hist[:,:,3:])
        
        vel = self.vel_encoder(short_hist)

        actor_input = torch.cat([short_hist,vel,long_hist],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def VelLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        pred_vel = self.vel_encoder(obs_hist[:,-5:,3:].reshape(b,-1))
        loss = F.mse_loss(pred_vel,obs_hist[:,-1,:3].detach())
        
        return loss
    
class MlpBaselineBarlowRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBaselineBarlowRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        self.cnn_encoder = LongHistCnn(num_obs=num_prop-3,
                                       in_channels=num_hist,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.encoder = nn.Sequential(nn.Linear((num_prop-6) * 5,128),
                                     nn.BatchNorm1d(128),
                                     nn.ELU(),
                                     nn.Linear(128,64),
                                     nn.BatchNorm1d(64),
                                     nn.ELU(),
                                     nn.Linear(64,latent_dim))
        
        self.predict_encoder = nn.Sequential(nn.Linear((num_prop-6) * 5,128),
                                     nn.BatchNorm1d(128),
                                     nn.ELU(),
                                     nn.Linear(128,64),
                                     nn.BatchNorm1d(64),
                                     nn.ELU(),
                                     nn.Linear(64,32))
        
        self.vel_predict = nn.Sequential(nn.Linear(32,32),
                                     nn.ELU(),
                                     nn.Linear(32,3))
        
        self.latent_predict = nn.Sequential(nn.Linear(32,32),
                                     nn.ELU(),
                                     nn.Linear(32,latent_dim))
    
        actor_list = mlp_factory(activation=activation,
                                 input_dims= (num_prop-3) * 5 + 64 + 3 + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        short_hist = obs_hist[:,-5:,3:].reshape(b,-1)
        short_est_hist = obs_hist[:,-5:,6:].reshape(b,-1)
        long_hist = self.cnn_encoder(obs_hist[:,:,3:])
        
        with torch.no_grad():
            z = self.predict_encoder(short_est_hist)
            vel = self.vel_predict(z)
            latent = self.latent_predict(z)
            
        actor_input = torch.cat([short_hist,vel.detach(),latent.detach(),long_hist],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        z1 = self.encoder(obs_hist[:,-5:,6:].reshape(b,-1))
        z2 = self.encoder(obs_hist[:,-6:-1,6:].reshape(b,-1))

        z1_l = self.projector(z1) 
        z2_l = self.projector(z2)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_encoder(obs_hist[:,-6:-1,6:].reshape(b,-1))
        pred_z_l = self.latent_predict(pred_z)
        pred_vel = self.vel_predict(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
"""
class MlpBVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAEActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = BetaVAE(in_dim=5*(num_prop-8),beta=0.1,output_dim=num_prop-8) #remove baselin and command
        # self.obs_normalizer = EmpiricalNormalization(shape=num_prop)
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,5:,8:].reshape(b,-1)) #remove linvel and command
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        # obs_hist = obs_hist[:,5:,:].view(b,-1)
        recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,4:-1,8:].reshape(b,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,-1,8:],recon,mu,log_var) # remove linvel and command
        # mseloss = F.mse_loss(predicted_vel,priv)
        mseloss = F.mse_loss(predicted_vel,0.1*obs_hist[:,-2,:3].detach())

        loss = loss + mseloss
        return loss
    
"""
    
# class MlpBaselineVAEActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  num_actions,
#                  actor_dims,
#                  latent_dim,
#                  activation) -> None:
#         super(MlpBaselineVAEActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
        
        
#         self.cnn_encoder = LongHistCnn(num_obs=num_prop-3,
#                                        in_channels=num_hist,
#                                        output_size=64,
#                                        kernel_size=[6, 4],
#                                        filter_size=[32, 16],
#                                        stride_size=[3, 2])
        
#         self.Vae = BetaVAE(in_dim=5*(num_prop-9),beta=0.04,output_dim=num_prop-9) #remove baselin and command
    
#         actor_list = mlp_factory(activation=activation,
#                                  input_dims= (num_prop-3) * 5 + 64 + 3 + 16,
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims)
#         self.actor = nn.Sequential(*actor_list)
#         print(self.actor)
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_hist = obs_hist[:,-5:,3:].reshape(b,-1)
#         short_est_hist = obs_hist[:,-5:,9:].reshape(b,-1)
#         long_hist = self.cnn_encoder(obs_hist[:,:,3:])
        
#         with torch.no_grad():
#             z,vel = self.Vae.get_latent(short_est_hist) #remove linvel and command
            
#         actor_input = torch.cat([short_hist,vel.detach(),z.detach(),long_hist],dim=-1) 
#         mean  = self.actor(actor_input)
#         return mean
        
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,-6:-1,9:].reshape(b,-1)) # remove linvel and command
#         loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,mu,log_var) # remove linvel and command
#         mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())

#         loss = loss + mseloss
        
#         return loss

class MlpBaselineVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBaselineVAEActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        self.cnn_encoder = LongHistCnn(num_obs=num_prop-3,
                                       in_channels=num_hist,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.Vae = BetaVAE(in_dim=5*(num_prop-8),beta=0.04,output_dim=num_prop-8) #remove baselin and command
    
        actor_list = mlp_factory(activation=activation,
                                 input_dims= (num_prop-3) * 5 + 64 + 3 + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        short_hist = obs_hist[:,-5:,3:].reshape(b,-1)
        short_est_hist = obs_hist[:,-5:,8:].reshape(b,-1)
        long_hist = self.cnn_encoder(obs_hist[:,:,3:])
        
        with torch.no_grad():
            z,vel = self.Vae.get_latent(short_est_hist) #remove linvel and command
            
        actor_input = torch.cat([short_hist,vel.detach(),z.detach(),long_hist],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,-6:-1,8:].reshape(b,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,-1,8:],recon,mu,log_var) # remove linvel and command
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())

        loss = loss + mseloss
        
        return loss

# class MlpBaselineVAEActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  num_actions,
#                  actor_dims,
#                  latent_dim,
#                  activation) -> None:
#         super(MlpBaselineVAEActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
        
        
#         self.cnn_encoder = LongHistCnn(num_obs=num_prop-8,
#                                        in_channels=num_hist,
#                                        output_size=32,
#                                        kernel_size=[6, 4],
#                                        filter_size=[32, 16],
#                                        stride_size=[3, 2])
        
#         self.Vae = BetaVAE(in_dim=5*(num_prop-8),beta=0.04,output_dim=num_prop-8) #remove baselin and command
        
#         self.mlp_encoder = nn.Sequential(nn.Linear((num_prop-8) * 5,128),
#                                           nn.ELU(),
#                                           nn.Linear(128,64),
#                                           nn.ELU(),
#                                           nn.Linear(64,16))
    
#         actor_list = mlp_factory(activation=activation,
#                                  input_dims= num_prop + 16 + 32 + 16,
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims)
#         self.actor = nn.Sequential(*actor_list)
#         print(self.actor)
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_hist = obs_hist[:,-5:,3:].reshape(b,-1)
#         short_est_hist = obs_hist[:,-5:,8:].reshape(b,-1)
#         long_hist = self.cnn_encoder(obs_hist[:,:,3:])
        
#         with torch.no_grad():
#             z,vel = self.Vae.get_latent(short_est_hist) #remove linvel and command
            
#         actor_input = torch.cat([short_hist,vel.detach(),z.detach(),long_hist],dim=-1) 
#         mean  = self.actor(actor_input)
#         return mean
        
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,-6:-1,8:].reshape(b,-1)) # remove linvel and command
#         loss = self.Vae.loss_fn(obs_hist[:,-1,8:],recon,mu,log_var) # remove linvel and command
#         mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())

#         loss = loss + mseloss
        
#         return loss

# class MlpBaselineVAEActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  num_actions,
#                  actor_dims,
#                  latent_dim,
#                  activation) -> None:
#         super(MlpBaselineVAEActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
        
        
#         self.cnn_encoder = LongHistCnn(num_obs=num_prop-3,
#                                        in_channels=num_hist,
#                                        output_size=64,
#                                        kernel_size=[6, 4],
#                                        filter_size=[32, 16],
#                                        stride_size=[3, 2])
        
#         self.Vae = BetaVAE(in_dim=5*(num_prop-3),beta=0.2,output_dim=num_prop-3) #remove baselin and command
    
#         actor_list = mlp_factory(activation=activation,
#                                  input_dims= (num_prop-3) * 5 + 64 + 3 + 16,
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims)
#         self.actor = nn.Sequential(*actor_list)
#         print(self.actor)
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_hist = obs_hist[:,-5:,3:].reshape(b,-1)
#         long_hist = self.cnn_encoder(obs_hist[:,:,3:])
        
#         with torch.no_grad():
#             z,vel = self.Vae.get_latent(short_hist) #remove linvel and command
            
#         actor_input = torch.cat([short_hist,vel.detach(),z.detach(),long_hist],dim=-1) 
#         mean  = self.actor(actor_input)
#         return mean
        
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,-6:-1,3:].reshape(b,-1)) # remove linvel and command
#         loss = self.Vae.loss_fn(obs_hist[:,-1,3:],recon,mu,log_var) # remove linvel and command
#         mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())

#         loss = loss + mseloss
        
#         return loss

class MlpBarlowTwinsNewCnnRegressionNoPhaseActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsNewCnnRegressionNoPhaseActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = LongHistCnn(num_obs=num_prop-6,
                                       in_channels=num_hist-1,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = LongHistCnn(num_obs=num_prop-6,
                                       in_channels=num_hist-1,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        actor_list = mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop - 3)*5 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,6:]) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-5:,3:].reshape(b,-1)],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,6:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,6:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,6:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpBarlowTwinsNewCnnRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsNewCnnRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        self.cnn_encoder = LongHistCnn(num_obs=num_prop-8,
                                       in_channels=num_hist-1,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        self.predict_cnn_encoder = LongHistCnn(num_obs=num_prop-8,
                                       in_channels=num_hist-1,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        actor_list = mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-3)*5 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,8:]) #remove linvel and command
            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-5:,3:].reshape(b,-1)],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,8:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,8:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,8:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpBaselineTransActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBaselineTransActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        self.trans_encoder = StateCausalTransformer(n_obs=num_prop-3,
                                                     n_embd=64,
                                                     n_head=4,
                                                     n_layer=1,
                                                     dropout=0,
                                                     block_size=10,
                                                     bias=False)
        
        self.vel_est = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        actor_list = mlp_factory(activation=activation,
                                 input_dims= 64 + 3 ,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        long_hist = self.trans_encoder(obs_hist[:,:,3:])[:,-1,:]
        vel = self.vel_est(long_hist)
        
        mean  = self.actor(torch.cat([long_hist,vel],dim=-1))
        return mean
        
    def VelLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        long_hist = self.trans_encoder(obs_hist[:,:,3:])[:,-1,:]
        vel = self.vel_est(long_hist)
        mseloss = F.mse_loss(vel,obs_hist[:,-1,:3].detach())
        
        return mseloss
    
class MlpBaselineRnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBaselineRnnActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        self.trans_encoder = StateCausalTransformer(n_obs=num_prop-3,
                                                     n_embd=64,
                                                     n_head=4,
                                                     n_layer=1,
                                                     dropout=0,
                                                     block_size=10,
                                                     bias=False)
        
        self.vel_est = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        actor_list = mlp_factory(activation=activation,
                                 input_dims= 64 + 3 ,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        long_hist = self.trans_encoder(obs_hist[:,:,3:])[:,-1,:]
        vel = self.vel_est(long_hist)
        
        mean  = self.actor(torch.cat([long_hist,vel],dim=-1))
        return mean
        
    def VelLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        long_hist = self.trans_encoder(obs_hist[:,:,3:])[:,-1,:]
        vel = self.vel_est(long_hist)
        mseloss = F.mse_loss(vel,obs_hist[:,-1,:3].detach())
        
        return mseloss

class MlpBaselineTerrianGuideActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBaselineTerrianGuideActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.height_size = 187
        self.prop_start = 3
        self.prop_nocmd_start = 8
        
        
        self.cnn_encoder = LongHistCnn(num_obs=num_prop-self.height_size-self.prop_start,
                                       in_channels=num_hist,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.height_scan_encoder = nn.Sequential(nn.Linear(self.height_size,128),
                                   nn.ELU(),
                                   nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64,32))
        self.vel_encoder = nn.Sequential(nn.Linear((num_prop-self.height_size-self.prop_start) * 5,128),
                                   nn.ELU(),
                                   nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64,3))
        
        self.dropout = nn.Dropout(p=0.0)
    
        actor_list = mlp_factory(activation=activation,
                                 input_dims= (num_prop-self.height_size-self.prop_start) * 5 + 64 + 3 + 32,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        self.dropout.p = 1 - self.random
        
        short_hist = obs_hist[:,-5:,self.height_size+self.prop_start:].reshape(b,-1)
        long_hist = self.cnn_encoder(obs_hist[:,:,self.height_size+self.prop_start:])
        vel = self.vel_encoder(short_hist)
        
        height_latent = self.dropout(self.height_scan_encoder(obs_hist[:,-1,:self.height_size]))
            
        actor_input = torch.cat([short_hist,vel,height_latent,long_hist],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def VelLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        predicted_vel  = self.vel_encoder(obs_hist[:,-5:,self.height_size+self.prop_start:].reshape(b,-1)) # remove linvel and command
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-1,:3].detach())
        
        return mseloss
    
class MlpBaselineVQVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBaselineVQVAEActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        self.cnn_encoder = LongHistCnn(num_obs=num_prop-3,
                                       in_channels=num_hist,
                                       output_size=64,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
        
        self.Vae = VQVAE_vel(in_dim=5*(num_prop-3),output_dim=(num_prop-3),num_emb=64) #remove baselin and command
    
        actor_list = mlp_factory(activation=activation,
                                 input_dims= (num_prop-3) * 5 + 64 + 3 + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims)
        self.actor = nn.Sequential(*actor_list)
        print(self.actor)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        short_hist = obs_hist[:,-5:,3:].reshape(b,-1)
        long_hist = self.cnn_encoder(obs_hist[:,:,3:])
        
        with torch.no_grad():
            z,vel = self.Vae.get_latent(short_hist) #remove linvel and command
            
        actor_input = torch.cat([short_hist,vel.detach(),z.detach(),long_hist],dim=-1) 
        mean  = self.actor(actor_input)
        return mean
        
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        recon,quantize,z,predicted_vel  = self.Vae(obs_hist[:,-6:-1,3:].reshape(b,-1)) # remove linvel and command
        loss =  self.Vae.loss_fn(obs_hist[:,-1,3:],recon,quantize,z) # remove linvel and command
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())

        loss = loss + mseloss
        
        return loss

# class VQVAE_CNN_vel(nn.Module):

#     def __init__(self,
#                  in_dim= 45*5,
#                  latent_dim = 16,
#                  encoder_hidden_dims = [128,64],
#                  decoder_hidden_dims = [64,128],
#                  output_dim = 45,
#                  num_emb = 32) -> None:
        
#         super(VQVAE_CNN_vel, self).__init__()

#         self.latent_dim = latent_dim
        
#         encoder_layers = []
#         encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(encoder_hidden_dims)-1):
#             encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
#                                         nn.ELU()))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
#         self.fc_vel = nn.Sequential(nn.Linear(encoder_hidden_dims[-1], 3))

#         # Build Decoder
#         decoder_layers = []
#         decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
#             else:
#                 decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
#                                       nn.ELU()))

#         self.decoder = nn.Sequential(*decoder_layers)
#         self.embedding_dim = latent_dim
#         self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=num_emb)
    
#     def get_latent(self,input):
#         z,vel = self.encode(input)
#         z = self.quantizer(z)
#         return z,vel

#     def encode(self,input):
        
#         latent = self.encoder(input)
#         z = self.fc_mu(latent)
#         #z = F.normalize(z,dim=-1,p=2)
#         z = F.normalize(z)
#         vel = self.fc_vel(latent)
#         return z,vel 
    
#     def decode(self,quantized,z):
#         quantized = z + (quantized - z).detach()
#         input_hat = self.decoder(quantized)
#         return input_hat
    
#     def forward(self, input):
#         z,vel = self.encode(input)
#         quantize = self.quantizer(z)
#         input_hat = self.decode(quantize,z)
#         return input_hat,quantize,z,vel
    
#     def loss_fn(self,y, y_hat,quantized,z):
#         recon_loss = F.mse_loss(y_hat, y)
        
#         commitment_loss = F.mse_loss(
#             quantized.detach(),
#             z
#         )

#         embedding_loss = F.mse_loss(
#             quantized,
#             z.detach()
#         )

#         vq_loss = 0.25*commitment_loss + embedding_loss

#         return recon_loss + vq_loss

# class MlpVQVAEMixedActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpVQVAEMixedActor,self).__init__()
#         self.num_hist = num_hist
#         self.num_prop = num_prop

#         # self.actor = nn.Sequential(*mlp_factory(activation=activation,
#         #                          input_dims=latent_dim + num_prop,
#         #                          out_dims=num_actions,
#         #                          hidden_dims=actor_dims))
        
#         self.actor = MixedMlp(input_size=latent_dim+num_prop,
#                               latent_size=9+latent_dim,
#                               hidden_size=128,
#                               num_actions=num_actions,
#                               num_experts=4)
       
#         self.Vae = VQVAE_vel(in_dim=5*(num_prop-9),output_dim=(num_prop-9),num_emb=64)
    
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random
        
#     def normalize_reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten.detach()
#         #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
#         return obs_hist

        
#     def forward(self,obs_hist_flatten):
#         obs_hist = self.normalize_reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             latents,predicted_vel = self.Vae.get_latent(obs_hist[:,-5:,9:].reshape(b,-1))

#         actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,9:].detach()],dim=-1)
        
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.normalize_reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()

#         recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,-6:-1,9:].reshape(b,-1))
#         loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z) 
#         mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
#         loss = loss + mseloss

#         return loss

class MlpVQVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAEActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = VQVAE_vel(in_dim=5*(num_prop-9),output_dim=(num_prop-9),num_emb=128)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random
        
    def normalize_reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist

        
    def forward(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,-5:,9:].reshape(b,-1))

        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,-6:-1,9:].reshape(b,-1))
        loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z) 
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
        loss = loss + mseloss

        return loss
    
class MlpVQVAEShortHistActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAEShortHistActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.short_hist_encode = nn.Sequential(nn.Linear((num_prop-9) * 5,128),
                                   nn.ELU(),
                                   nn.Linear(128,64),
                                   nn.ELU(),
                                   nn.Linear(64,16))
       
        self.Vae = VQVAE_vel(in_dim=5*(num_prop-9),output_dim=(num_prop-9),num_emb=64)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random
        
    def normalize_reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist

        
    def forward(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,-5:,9:].reshape(b,-1))

        short_hist_latent = F.normalize(self.short_hist_encode(obs_hist[:,-5:,9:].reshape(b,-1)))
        actor_input = torch.cat([short_hist_latent,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,-6:-1,9:].reshape(b,-1))
        loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z) 
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
        loss = loss + mseloss

        return loss
    
class MlpVQVAELongHistActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAELongHistActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.long_hist_encode = LongHistCnn(num_obs=num_prop-9,
                                       in_channels=num_hist,
                                       output_size=16,
                                       kernel_size=[6, 4],
                                       filter_size=[32, 16],
                                       stride_size=[3, 2])
       
        self.Vae = VQVAE_vel(in_dim=5*(num_prop-9),output_dim=(num_prop-9),num_emb=64)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random
        
    def normalize_reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist

        
    def forward(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,-5:,9:].reshape(b,-1))

        long_hist_latent = F.normalize(self.long_hist_encode(obs_hist[:,:,9:]))
        actor_input = torch.cat([long_hist_latent,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,-6:-1,9:].reshape(b,-1))
        loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z) 
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
        loss = loss + mseloss

        return loss
    
class MlpVQVAERnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAERnnActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = VQVAE_EMA_Rnn(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=64)
        
    def normalize_reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist

        
    def forward(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,:,9:])

        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        recon,quantize,z,predicted_vel,onehot_encode = self.Vae(obs_hist[:,:-1,9:])
        loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z,onehot_encode) 
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
        loss = loss + mseloss

        return loss

class VQVAE_NoEncode(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 num_emb = 32) -> None:
        
        super(VQVAE_NoEncode, self).__init__()

        self.latent_dim = latent_dim

        self.fc_mu = nn.Sequential(nn.Linear(in_dim,64),
                                   nn.ELU(),
                                   nn.Linear(64,64),
                                   nn.ELU(),
                                   nn.Linear(64,16))
        self.fc_vel = nn.Sequential(nn.Linear(in_dim,64),
                                   nn.ELU(),
                                   nn.Linear(64,64),
                                   nn.ELU(),
                                   nn.Linear(64,3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=num_emb)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        z = self.quantizer(z)
        return z,vel

    def encode(self,input):
        
        z = self.fc_mu(input)
        #z = F.normalize(z,dim=-1,p=2)
        z = F.normalize(z)
        vel = self.fc_vel(input)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss

class MlpVQVAERnnEncodeActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAERnnEncodeActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + 3 + 64 + 6,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.rnn_encoder = RnnStateHistoryEncoder(activation_fn=nn.ELU(),
                                              input_size=num_prop-9,
                                              mlp_output_size=64,
                                              encoder_dims=[64],
                                              hidden_size=64)
       
        self.Vae = VQVAE_NoEncode(in_dim=64,output_dim=(num_prop-9),num_emb=64)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random
        
    def normalize_reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist

        
    def forward(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        rnn_encode = self.rnn_encoder(obs_hist[:,:,9:])
        
        # with torch.no_grad():
        latents,predicted_vel = self.Vae.get_latent(rnn_encode)

        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),rnn_encode,obs_hist[:,-1,3:9]],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        # with torch.no_grad():
        rnn_encode = self.rnn_encoder(obs_hist[:,:-1,9:])
        recon,quantize,z,predicted_vel = self.Vae(rnn_encode.detach())
        loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z) 
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
        loss = loss + mseloss

        return loss
    
class MlpVQVAEMixedActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAEMixedActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop

        # self.actor = nn.Sequential(*mlp_factory(activation=activation,
        #                          input_dims=latent_dim + num_prop,
        #                          out_dims=num_actions,
        #                          hidden_dims=actor_dims))
        
        self.actor_mixed = MixedMlp(input_size=num_prop-3,
                                    latent_size=latent_dim+3,
                                    hidden_size=128,
                                    num_actions=num_actions,
                                    num_experts=4)
       
        self.Vae = VQVAE_vel(in_dim=5*(num_prop-9),output_dim=(num_prop-9),num_emb=64)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random
        
    def normalize_reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist

        
    def forward(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,-5:,9:].reshape(b,-1))

        latent_input = torch.cat([latents.detach(),predicted_vel.detach()],dim=-1)
        mean  = self.actor_mixed(latent_input,obs_hist[:,-1,3:].detach())
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,-6:-1,9:].reshape(b,-1))
        loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z) 
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
        loss = loss + mseloss

        return loss
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ELU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ELU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2,)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, L_in, C_in)"""
        y1 = self.tcn(inputs.transpose(1,2))  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return o
    

class MlpBarlowTwinsLongCnnRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsLongCnnRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        # barlow related
        # self.cnn_encoder = TCN(input_size=num_prop-9,
        #                        output_size=32,
        #                        num_channels=[32,32,32],
        #                        kernel_size=5)
        
        self.cnn_encoder = StateHistoryEncoder(input_size=num_prop-9,
                               output_size=64,
                               tsteps=50,
                               activation_fn=nn.ELU)
        
        
        self.latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        # future related 
        # self.predict_cnn_encoder = TCN(input_size=num_prop-9,
        #                                output_size=32,
        #                                num_channels=[32,32,32],
        #                                kernel_size=5)
        
        self.predict_cnn_encoder = StateHistoryEncoder(input_size=num_prop-9,
                               output_size=64,
                               tsteps=50,
                               activation_fn=nn.ELU)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
    def set_random(self,random):
        pass
    
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.predict_cnn_encoder(obs_hist[:,1:,9:]) #remove linvel and command

            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1 = self.cnn_encoder(obs_hist[:,1:,9:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,9:])

        z1_l_ = self.latent_layer(z1)
        z2_l_ = self.latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_z = self.predict_cnn_encoder(obs_hist[:,:-1,9:])
        pred_z_l = self.predict_latent_layer(pred_z)
        pred_vel = self.predict_vel_layer(pred_z)

        latent_loss = F.mse_loss(pred_z_l,z1_l_)
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss + latent_loss
        
        return loss
    
class MlpBarlowTwinsCnnSingleActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 actor_dims,
                 latent_dim,
                 activation) -> None:
        super(MlpBarlowTwinsCnnSingleActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        
        self.cnn_encoder = CnnHistoryEncoder(input_size=num_prop-9,
                                             tsteps = 20)
        
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.single_encoder = nn.Sequential(
                                          nn.Linear(num_prop-9,64),
                                          nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        self.bn = nn.BatchNorm1d(64,affine=False)
        
    def set_random(self,random):
        pass
    
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten#.detach()
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.cnn_encoder(obs_hist[:,1:,9:]) #remove linvel and command

            z = self.predict_latent_layer(latent)
            vel = self.predict_vel_layer(latent)
            
        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
        
    def BarlowTwinsLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z1_l_ = self.single_encoder(obs_hist[:,-1,9:])
        z2 = self.cnn_encoder(obs_hist[:,:-1,9:])

        z2_l_ = self.predict_latent_layer(z2)

        z1_l = self.projector(z1_l_) 
        z2_l = self.projector(z2_l_)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        pred_vel = self.predict_vel_layer(z2)
        
        priv_loss = F.mse_loss(pred_vel,obs_hist[:,-2,:3].detach())

        loss = on_diag + 5e-3*off_diag + priv_loss
        
        return loss
    
class VQVAE_CNN(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 num_emb = 32) -> None:
        
        super(VQVAE_CNN, self).__init__()

        self.latent_dim = latent_dim
        
        self.encoder = TCN(input_size=in_dim,
                               output_size=32,
                               num_channels=[32,32,32],
                               kernel_size=2)

        self.fc_mu = nn.Sequential(nn.Linear(32, 32),
                                   nn.ELU(),
                                   nn.Linear(32,latent_dim))
        
        self.fc_vel = nn.Sequential(nn.Linear(32, 32),
                                    nn.ELU(),
                                    nn.Linear(32, 3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=num_emb)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        z = self.quantizer(z)
        return z,vel

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        z = F.normalize(z)
        vel = self.fc_vel(latent)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss
    
class VQVAE_EMA_Cnn(nn.Module):

    def __init__(self,
                 in_dim= 45,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 num_emb=32) -> None:
        
        super(VQVAE_EMA_Cnn, self).__init__()

        self.latent_dim = latent_dim
        
        self.encoder = TCN(input_size=in_dim,
                               output_size=32,
                               num_channels=[32,32,32],
                               kernel_size=2)

        self.fc_mu = nn.Sequential(nn.Linear(32, 32),
                                   nn.ELU(),
                                   nn.Linear(32,latent_dim))
        
        self.fc_vel = nn.Sequential(nn.Linear(32, 32),
                                    nn.ELU(),
                                    nn.Linear(32, 3))

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=num_emb)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        return z,vel

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        z = F.normalize(z)
        vel = self.fc_vel(latent)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize,onehot_encode = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel,onehot_encode
    
    def loss_fn(self,y, y_hat,quantized,z,onehot_encode):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )
        self.quantizer.update_codebook(z,onehot_encode)

        vq_loss = 0.25*commitment_loss #+ embedding_loss

        return recon_loss + vq_loss
    
class MlpVQVAECnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAECnnActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = VQVAE_CNN(in_dim=num_prop-9,output_dim=(num_prop-9),num_emb=128)
    
        self.random = 1
        
    def set_random(self,random):
        self.random = random
        
    def normalize_reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist

        
    def forward(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist[:,1:,9:])

        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.normalize_reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,:-1,9:])
        loss = self.Vae.loss_fn(obs_hist[:,-1,9:],recon,quantize,z) 
        mseloss = F.mse_loss(predicted_vel,obs_hist[:,-2,:3].detach())
        loss = loss + mseloss

        return loss

class MlpSimSiamActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(MlpSimSiamActor,self).__init__()
        self.num_hist = num_hist
        self.num_prop = num_prop
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop + latent_dim,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

        # past encoder
        self.encoder = StateHistoryEncoder(input_size=num_prop-9,
                               output_size=64,
                               tsteps=20,
                               activation_fn=nn.ReLU)
               
        self.latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Sequential(nn.Linear(64,32),
                                       nn.ELU(),
                                       nn.Linear(32,3))
        # future encoder
        self.future_encoder = nn.Sequential(nn.Linear(num_prop-9,128),
                                    nn.BatchNorm1d(128),
                                    nn.ELU(),
                                    nn.Linear(128,64),
                                    nn.BatchNorm1d(64),
                                    nn.ELU(),
                                    nn.Linear(64,latent_dim))
        
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(16, 8),
                                        nn.BatchNorm1d(8),
                                        nn.ELU(), # hidden layer
                                        nn.Linear(8, 16)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1)
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random
        
    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist_flatten = obs_hist_flatten.detach()
        #obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)
        return obs_hist
        
    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            latent = self.encoder(obs_hist[:,1:,9:])
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)

        actor_input = torch.cat([z.detach(),vel.detach(),obs_hist[:,-1,3:].detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def SimSiamLoss(self,obs_hist_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        z2 = self.future_encoder(obs_hist[:,-1,9:])
        z1_ = self.encoder(obs_hist[:,:-1,9:])

        z1_vel = self.vel_layer(z1_)
        z1 = self.latent_layer(z1_)
        
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        priv_loss = F.mse_loss(z1_vel,obs_hist[:,-2,:3].detach())
        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5 
        loss = loss + priv_loss 
        return loss

class PureBetaVAE(nn.Module):

    def __init__(self,
                 in_dim= 45,
                 latent_dim = 16,
                 encoder_hidden_dims = [64,32],
                 decoder_hidden_dims = [32,64],
                 output_dim = 45,
                 beta: int = 0.1) -> None:
        
        super(PureBetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),

                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)

        self.kl_weight = beta

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu,log_var]
    
    def get_latent(self,input):
        mu,log_var = self.encode(input)
        return mu

    def decode(self,z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        
        mu,log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z),z, mu, log_var]
    
    def loss_fn(self,y, y_hat, mean, logvar):

        recons_loss = F.mse_loss(y_hat,y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), -1))
        loss = recons_loss + self.beta * kl_loss

        return loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureTeacherVAE(nn.Module):
    """
    Posterior = Prior + Residual：
      prior_encode(obs) -> (mu_p, logvar_p)
      post_encode(priv, obs) -> (mu_q, logvar_q) 其中
        μ_q = μ_p + Δμ,  logσ_q^2 = logσ_p^2 + Δlogσ^2
    forward 返回 z 以及两侧分布参数；
    loss_fn 计算 KL(q||p) + w_nce * InfoNCE。
    """
    def __init__(self,
                 in_dim=45,
                 post_indim=187,
                 latent_dim=16,
                 encoder_hidden_dims=[128, 64],
                 res_hidden_dims=[64, 32],
                 beta: float = 0.1,
                 # ---- InfoNCE 相关超参 ----
                 nce_proj_dim: int = 128,
                 nce_queue_size: int = 32768,
                 nce_temperature: float = 0.1,
                 nce_weight: float = 0.5) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # ---- Prior encoder: obs -> (mu_p, logvar_p)
        enc_layers = [nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]), nn.ELU())]
        for i in range(len(encoder_hidden_dims) - 1):
            enc_layers += [nn.Sequential(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i+1]), nn.ELU())]
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu     = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # ---- Residual encoder: priv -> (Δμ, Δlogσ^2)
        res_layers = [nn.Sequential(nn.Linear(post_indim, res_hidden_dims[0]), nn.ELU())]
        for i in range(len(res_hidden_dims) - 1):
            res_layers += [nn.Sequential(nn.Linear(res_hidden_dims[i], res_hidden_dims[i+1]), nn.ELU())]
        self.res_encoder = nn.Sequential(*res_layers)
        self.res_mu     = nn.Linear(res_hidden_dims[-1], latent_dim)
        self.res_logvar = nn.Linear(res_hidden_dims[-1], latent_dim)

        # ---- InfoNCE: 投影头 + 队列 ----
        self.nce_q_proj = nn.Sequential(
            nn.Linear(latent_dim, nce_proj_dim), nn.ELU(),
            nn.Linear(nce_proj_dim, nce_proj_dim))
        self.nce_k_proj = nn.Sequential(
            nn.Linear(latent_dim, nce_proj_dim), nn.ELU(),
            nn.Linear(nce_proj_dim, nce_proj_dim))

        self.nce_temperature = nce_temperature
        self.nce_weight = nce_weight

        # 负样本队列（[N, P]），注册为 buffer
        queue = torch.randn(nce_queue_size, nce_proj_dim)
        queue = F.normalize(queue, dim=1)
        self.register_buffer("nce_queue", queue)
        self.register_buffer("nce_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.nce_queue_size = nce_queue_size
        self.nce_proj_dim = nce_proj_dim

    # ---------- helpers ----------
    @staticmethod
    def _reparameterize(mu, logvar, training: bool = True):
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    @staticmethod
    def _kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
        # D_KL( N(μq,σq^2) || N(μp,σp^2) )
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
        return kl.sum(dim=-1).mean()

    @torch.no_grad()
    def _nce_update_memory(self, k_proj_norm):
        """把当前 batch 的 key 投影放入队列（FIFO）。k_proj_norm: [B, P]"""
        bsz = k_proj_norm.shape[0]
        ptr = int(self.nce_queue_ptr.item())
        end = ptr + bsz
        if end <= self.nce_queue_size:
            self.nce_queue[ptr:end, :] = k_proj_norm
        else:
            first = self.nce_queue_size - ptr
            self.nce_queue[ptr:, :] = k_proj_norm[:first, :]
            self.nce_queue[:end - self.nce_queue_size, :] = k_proj_norm[first:, :]
        self.nce_queue_ptr[0] = (ptr + bsz) % self.nce_queue_size

    def _info_nce(self, q_feats, k_feats_detached):
        """
        q_feats: [B, D_latent] 或其投影的输入
        k_feats_detached: [B, D_latent]（外部需 detach）
        返回：InfoNCE 标量 loss
        """
        # 投影 + 归一化
        q = self.nce_q_proj(q_feats)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            k = self.nce_k_proj(k_feats_detached)
            k = F.normalize(k, dim=1)

        # 正样本相似度：q·k+
        l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(-1)  # [B,1]

        # 负样本相似度：q·queue^T
        queue = self.nce_queue.clone().detach()               # [N,P]
        l_neg = torch.einsum("bd,nd->bn", q, queue)           # [B,N]

        logits = torch.cat([l_pos, l_neg], dim=1) / self.nce_temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # 正样本在 index 0
        loss = F.cross_entropy(logits, labels)

        # 入队（用与 loss 中相同的 k）
        self._nce_update_memory(k)

        return loss
    # ---------------------------------------------

    def prior_encode(self, obs_input):
        h = self.encoder(obs_input)
        mu_p = self.fc_mu(h)
        logvar_p = torch.clamp(self.fc_logvar(h), -5.0, 5.0)
        return mu_p, logvar_p

    def post_encode(self, priv_input, obs_input):
        mu_p, logvar_p = self.prior_encode(obs_input)
        r = self.res_encoder(priv_input)
        d_mu     = self.res_mu(r)
        d_logvar = torch.clamp(self.res_logvar(r), -5.0, 5.0)
        mu_q     = mu_p + d_mu
        logvar_q = torch.clamp(logvar_p + d_logvar, -9.0, 5.0)
        return mu_q, logvar_q, mu_p, logvar_p

    def forward(self,
                obs_input,
                priv_input=None,
                sample_from: str = "post",  # 注意：你的字符串与判断分支要统一（"posterior"/"prior"）
                eval_use_mean: bool = True):
        if priv_input is not None:
            mu_q, logvar_q, mu_p, logvar_p = self.post_encode(priv_input, obs_input)
        else:
            mu_p, logvar_p = self.prior_encode(obs_input)
            mu_q, logvar_q = None, None

        if sample_from in ("post", "posterior") and (mu_q is not None):
            z = self._reparameterize(mu_q, logvar_q, training=self.training)
            if (not self.training) and eval_use_mean:
                z = mu_q
        else:
            # 与部署一致：prior
            z = self._reparameterize(mu_p, logvar_p, training=self.training)
            if (not self.training) and eval_use_mean:
                z = mu_p

        return z, (mu_p, logvar_p), (mu_q, logvar_q)

    def loss_fn(self,
                obs_input,
                priv_input,
                kl_weight: float = None,
                nce_weight: float = 1.0):
        """
        总损失 = beta*KL(q||p) + nce_weight * InfoNCE( prior(μ_p), posterior(μ_q) )
        - key (posterior) 停梯度
        - 若 priv_input 为 None，则只返回 KL（实际上无 posterior，KL 也无法算，这里假定训练期总有 priv）
        """
        mu_q, logvar_q, mu_p, logvar_p = self.post_encode(priv_input, obs_input)

        # KL(q||p)
        kl = self._kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
        w_kl = self.beta if kl_weight is None else kl_weight
        loss = w_kl * kl

        # InfoNCE（prior μ_p 为 query，posterior μ_q 为 key，stop-grad）
        w_nce = self.nce_weight if nce_weight is None else nce_weight
        if w_nce > 0:
            nce = self._info_nce(mu_p, mu_q.detach())
            loss = loss + w_nce * nce
        else:
            nce = torch.tensor(0.0, device=obs_input.device)

        
        return loss
    
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
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
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
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            short_encode = self.short_encoder(obs_hist[:,-5:,9:].reshape(b,-1))
            long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)


        actor_input = torch.cat([latents,predicted_contact.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,6:8]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)#critic_obs_flatten[:,:49]#torch.cat([height,obs_hist[:,-1,9:]],dim=-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) # remove linvel and command
        # recon,quantize,z,onehot_encode = self.Vae(vae_input)
        # loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(vae_input)
        
        short_encode = self.short_encoder(obs_hist[:,-6:-1,9:].reshape(b,-1))
        long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        encode = torch.cat([short_encode,long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-1,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_obs_flatten[:,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpBVAEDeltaRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAEDeltaRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-9,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            cnn_encode = self.predict_encoder(obs_hist[:,1:,9:]) #remove linvel and command
            current_latent = self.Vae.get_latent(obs_hist[:,-1,9:])
            latents = self.predict_latent_layer(cnn_encode) + current_latent
            predicted_vel = self.predict_vel_layer(cnn_encode)
        
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            future_latent = self.Vae.get_latent(obs_hist[:,-1,9:])
            current_latent = self.Vae.get_latent(obs_hist[:,-2,9:])
        
        cnn_encode = self.predict_encoder(obs_hist[:,:-1,9:])
        predict_latent = self.predict_latent_layer(cnn_encode) + current_latent.detach()
        predict_vel = self.predict_vel_layer(cnn_encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = loss + mseloss + latent_loss
        return loss
    
class MlpBVAEDeltaLatentHistRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAEDeltaLatentHistRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 4*latent_dim, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-9,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        with torch.no_grad():
            cnn_encode = self.predict_encoder(obs_hist[:,1:,9:]) #remove linvel and command
            current_latent = self.Vae.get_latent(obs_hist[:,-1,9:])
            past_four_latents = self.Vae.get_latent(obs_hist[:,-5:-1,9:])
            latents = self.predict_latent_layer(cnn_encode) + current_latent
            predicted_vel = self.predict_vel_layer(cnn_encode)
        
        actor_input = torch.cat([past_four_latents.reshape(b,-1),latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            future_latent = self.Vae.get_latent(obs_hist[:,-1,9:])
            current_latent = self.Vae.get_latent(obs_hist[:,-2,9:])
        
        cnn_encode = self.predict_encoder(obs_hist[:,:-1,9:])
        predict_latent = self.predict_latent_layer(cnn_encode) + current_latent.detach()
        predict_vel = self.predict_vel_layer(cnn_encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = loss + mseloss + latent_loss
        return loss
    
class MlpSimpleRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimpleRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 4*latent_dim, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.input_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-9,
                                             tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            cnn_encode = self.predict_encoder(obs_hist[:,1:,9:]) #remove linvel and command
            latents = self.predict_latent_layer(cnn_encode)
            predicted_vel = self.predict_vel_layer(cnn_encode)
            
        past_latents = self.input_encoder(obs_hist[:,-5:-1,9:]).reshape(b,-1)
         
        actor_input = torch.cat([past_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        #actor_input = torch.cat([past_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def RegLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        # regression
        with torch.no_grad():
            future_latent = self.input_encoder(obs_hist[:,-1,9:])
        cnn_encode = self.predict_encoder(obs_hist[:,:-1,9:])
        
        predict_latent = self.predict_latent_layer(cnn_encode)
        predict_vel = self.predict_vel_layer(cnn_encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = mseloss + latent_loss
        return loss
    
class MlpSimpleMlpRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimpleMlpRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 4*latent_dim, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.input_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,3))
        
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
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            encode = self.predict_encoder(obs_hist[:,-5:,9:].reshape(b,-1)) #remove linvel and command
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            
        past_latents = self.input_encoder(obs_hist[:,-5:-1,9:]).reshape(b,-1)
         
        actor_input = torch.cat([past_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        #actor_input = torch.cat([past_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def RegLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        # regression
        with torch.no_grad():
            future_latent = self.input_encoder(obs_hist[:,-1,9:])
        encode = self.predict_encoder(obs_hist[:,-6:-1,9:].reshape(b,-1))
        
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = mseloss + latent_loss
        return loss
    
# class MlpSimpleTempRegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpSimpleTempRegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop + 4*latent_dim, # remove baselin
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.input_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
#                                           nn.ELU(),
#                                           nn.Linear(128,64),
#                                           nn.ELU(),
#                                           nn.Linear(64,latent_dim))
        
#         self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-9,
#                                              tsteps = 20)
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(128+latent_dim,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
        
#         self.predict_vel_layer = nn.Sequential(nn.Linear(128+latent_dim,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_encode = self.input_encoder(obs_hist[:,-5:,9:].reshape(b,-1))
        
#         with torch.no_grad():
#             long_encode = self.predict_encoder(obs_hist[:,1:,9:]) #remove linvel and command
#             encode = torch.cat([short_encode,long_encode],dim=-1)
            
#             latents = self.predict_latent_layer(encode)
#             predicted_vel = self.predict_vel_layer(encode)
             
#         actor_input = torch.cat([short_encode,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
    
#     def RegLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,l,_ = obs_hist.size()
#         # regression
#         with torch.no_grad():
#             future_latent = self.input_encoder(obs_hist[:,-5:,9:].reshape(b,-1))
            
#         cnn_encode = self.predict_encoder(obs_hist[:,:-1,9:])
        
#         predict_latent = self.predict_latent_layer(cnn_encode)
#         predict_vel = self.predict_vel_layer(cnn_encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent.detach())
#         mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

#         loss = mseloss + latent_loss
#         return loss

# class MlpBVAEDeltaRegressionPhaseGateActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpBVAEDeltaRegressionPhaseGateActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop, # remove baselin
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
#         self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-9,
#                                              tsteps = 20)
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
#         self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.phase_gate = nn.Sequential(nn.Linear(latent_dim+num_prop,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,2),
#                                    nn.Sigmoid())
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             cnn_encode = self.predict_encoder(obs_hist[:,1:,9:]) #remove linvel and command
#             current_latent = self.Vae.get_latent(obs_hist[:,-1,9:])
#             latents = self.predict_latent_layer(cnn_encode) + current_latent
#             predicted_vel = self.predict_vel_layer(cnn_encode)
        
#         actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,l,_ = obs_hist.size()
        
#         # VAE update
#         recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
#         loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
#         # regression
#         with torch.no_grad():
#             future_latent = self.Vae.get_latent(obs_hist[:,-1,9:])
#             current_latent = self.Vae.get_latent(obs_hist[:,-2,9:])
        
#         cnn_encode = self.predict_encoder(obs_hist[:,:-1,9:])
#         predict_latent = self.predict_latent_layer(cnn_encode) + current_latent.detach()
#         predict_vel = self.predict_vel_layer(cnn_encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent.detach())
#         mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

#         loss = loss + mseloss + latent_loss
#         return loss


# class MlpBVAERegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpBVAERegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop, # remove baselin
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.Vae = PureBetaVAE(in_dim=num_prop-8,beta=0.2,output_dim=num_prop-8) #remove baselin and command
        
#         self.predict_encoder = CnnHistoryEncoder(input_size=num_prop-8,
#                                              tsteps = 20)
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
#         self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             cnn_encode = self.predict_encoder(obs_hist[:,1:,8:]) #remove linvel and command
#             latents = self.predict_latent_layer(cnn_encode)
#             predicted_vel = self.predict_vel_layer(cnn_encode)

#         actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,l,_ = obs_hist.size()
        
#         # VAE update
#         recon,z, mu, log_var = self.Vae(obs_hist[:,:,8:].reshape(b*l,-1)) # remove linvel and command
#         loss = self.Vae.loss_fn(obs_hist[:,:,8:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
#         # regression
#         with torch.no_grad():
#             _,_,future_latent,_ = self.Vae(obs_hist[:,-1,8:])
        
#         cnn_encode = self.predict_encoder(obs_hist[:,:-1,8:])
#         predict_latent = self.predict_latent_layer(cnn_encode)
#         predict_vel = self.predict_vel_layer(cnn_encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent)
#         mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

#         loss = loss + mseloss + latent_loss
#         return loss
    
class StateHistoryEncoder(nn.Module):
    def __init__(self, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()

        self.tsteps = tsteps
        self.output_shape = output_size

        if tsteps == 50:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), 
            nn.ELU(),
            nn.Linear(32,32),
            nn.ELU()
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.ELU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.ELU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.ELU(), nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), nn.ELU()
            )
        elif tsteps == 10:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), nn.ELU()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.ELU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.ELU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size),nn.ELU()
            )
        elif tsteps == 20:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), nn.ELU()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.ELU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.ELU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), nn.ELU()
            )
        else:
            raise NotImplementedError()

    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output

class MlpSimpleShortLongRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimpleShortLongRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 4*latent_dim - 2 + 64, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.input_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                                   tsteps = 50,
                                                   output_size=64)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        long_latents = self.predict_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        short_latents = self.input_encoder(obs_hist[:,-5:-1,9:]).reshape(b,-1)
        
        with torch.no_grad():
           
            latents = self.predict_latent_layer(long_latents)
            predicted_vel = self.predict_vel_layer(long_latents)
            
        #actor_input = torch.cat([past_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        actor_input = torch.cat([short_latents,long_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def RegLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        # regression
        with torch.no_grad():
            future_latent = self.input_encoder(obs_hist[:,-1,9:])
        cnn_encode = self.predict_encoder(obs_hist[:,:-1,9:])
        
        predict_latent = self.predict_latent_layer(cnn_encode)
        predict_vel = self.predict_vel_layer(cnn_encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = mseloss + latent_loss
        return loss
    
class RnnSlimStateHistoryEncoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RnnSlimStateHistoryEncoder,self).__init__()

        self.hidden_size = hidden_size
        
        self.rnn = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           num_layers = 1)
        
    def forward(self,obs):
        h_0 = torch.zeros(1,obs.size(0),self.hidden_size,device=obs.device).requires_grad_()
        out, h_n = self.rnn(obs,h_0)
        return out[:,-1,:]
    
class MlpSimpleRnnRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimpleRnnRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + 64 + 7, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.input_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_encoder = RnnSlimStateHistoryEncoder(input_size=latent_dim,hidden_size=64)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        history_latents = self.input_encoder(obs_hist[:,1:,9:])
        long_latents = self.predict_encoder(history_latents[:,1:,:]) #remove linvel and command
        
        with torch.no_grad():
           
            latents = self.predict_latent_layer(long_latents)
            predicted_vel = self.predict_vel_layer(long_latents)
            
        # actor_input = torch.cat([long_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        actor_input = torch.cat([long_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def RegLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        # regression
        with torch.no_grad():
            history_latents = self.input_encoder(obs_hist[:,:,9:])
            future_latent = history_latents[:,-1,:]
            
        rnn_encode = self.predict_encoder(history_latents[:,:-1,:])
        
        predict_latent = self.predict_latent_layer(rnn_encode)
        predict_vel = self.predict_vel_layer(rnn_encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = mseloss + latent_loss
        return loss
    
class MlpSimpleLongShortRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimpleLongShortRegressionActor,self).__init__()
        
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= latent_dim + 16 + 4*latent_dim + 7 + num_prop - 9, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        # short long encoder
        self.single_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                                tsteps=num_hist-1,
                                                output_size=16)
        # estimator 
        self.estimator_backbone = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_contact_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,2))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.random = 1
        
    def set_random(self,random):
        self.random = random

    def reshape(self,obs_hist_flatten):
        # N*(T*O) -> (N * T)* O -> N * T * O
        obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
        return obs_hist
    
    def reshape_critic(self,critic_obs_hist_flatten):
        height = critic_obs_hist_flatten[:,:187]
        critic_obs_hist = critic_obs_hist_flatten[:,187:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
        return critic_obs_hist,height

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        long_latents = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        short_latents = self.single_encoder(obs_hist[:,-5:-1,9:]).reshape(b,-1)
        
        with torch.no_grad():
            est_latent = self.estimator_backbone(obs_hist[:,-5:,9:].reshape(b,-1))
            predicted_latent = self.predict_latent_layer(est_latent)
            predicted_vel = self.predict_vel_layer(est_latent)
        actor_input = torch.cat([short_latents,long_latents,predicted_latent.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def RegLoss(self,obs_hist_flatten,critic_obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        critic_obs_hist,height = self.reshape_critic(critic_obs_hist_flatten)
        b,l,_ = obs_hist.size()
        # regression
        with torch.no_grad():
            future_latent =  self.single_encoder(obs_hist[:,-1,9:])
            
        est_latent = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        
        predict_latent = self.predict_latent_layer(est_latent)
        predict_vel = self.predict_vel_layer(est_latent)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = mseloss + latent_loss
        return loss
    
class MlpSimpleRnnPhaseShiftRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimpleRnnPhaseShiftRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 64, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.input_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_encoder = RnnSlimStateHistoryEncoder(input_size=latent_dim,hidden_size=64)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.phase_shift_layer = nn.Sequential(nn.Linear(64,32),
                                               nn.ELU(),
                                               nn.Linear(32,2),
                                               nn.Tanh())
        
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
        b,_,_ = obs_hist.size()
        
        history_latents = self.input_encoder(obs_hist[:,1:,9:])
        long_latents = self.predict_encoder(history_latents[:,1:,:]) #remove linvel and command
        
        phase_shift = self.phase_shift_layer(long_latents) + obs_hist[:,-1,6:8]
        with torch.no_grad():
           
            latents = self.predict_latent_layer(long_latents)
            predicted_vel = self.predict_vel_layer(long_latents)
            
        #actor_input = torch.cat([long_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        actor_input = torch.cat([long_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9],phase_shift],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def RegLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        # regression
        with torch.no_grad():
            history_latents = self.input_encoder(obs_hist[:,:,9:])
            future_latent = history_latents[:,-1,:]
            
        rnn_encode = self.predict_encoder(history_latents[:,:-1,:])
        
        predict_latent = self.predict_latent_layer(rnn_encode)
        predict_vel = self.predict_vel_layer(rnn_encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent.detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = mseloss + latent_loss
        return loss
    
class MlpSimpleLongShortLatentRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimpleLongShortLatentRegressionActor,self).__init__()
        
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= latent_dim + num_prop + 32 + 4*latent_dim - 2, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        # short long encoder
        self.single_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
                                          nn.ELU(),
                                          nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                                tsteps=num_hist-1,
                                                output_size=32)
        # estimator 
        self.estimator_backbone = nn.Sequential(nn.Linear(latent_dim*5,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        long_latents = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        short_latents = self.single_encoder(obs_hist[:,-5:,9:])
        
        with torch.no_grad():
            est_latent = self.estimator_backbone(short_latents.reshape(b,-1))
            predicted_latent = self.predict_latent_layer(est_latent)
            predicted_vel = self.predict_vel_layer(est_latent)
            
        # actor_input = torch.cat([long_latents,latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        actor_input = torch.cat([short_latents[:,:-1,:].reshape(b,-1),long_latents,predicted_latent.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        # actor_input = torch.cat([short_latents,long_latents,predicted_latent.detach(),predicted_vel.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def RegLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        # regression
        with torch.no_grad():
            short_latents =  self.single_encoder(obs_hist[:,-6:,9:])
            
        est_latent = self.estimator_backbone(short_latents[:,:-1,9:].reshape(b,-1))
        
        predict_latent = self.predict_latent_layer(est_latent)
        predict_vel = self.predict_vel_layer(est_latent)
        
        latent_loss = F.mse_loss(predict_latent,short_latents[:,-1,9:].detach())
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = mseloss + latent_loss
        return loss
    
class MlpBVAESlimRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAESlimRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.predict_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                          nn.ELU(),
                                          nn.Linear(128,64),
                                          nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,latent_dim))
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,3))
        
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
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            encode = self.predict_encoder(obs_hist[:,-5:,9:].reshape(b,-1)) #remove linvel and command
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)

        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.predict_encoder(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = loss + mseloss + latent_loss
        return loss

# class MlpVQVAESlimRegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpBVAESlimRegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + num_prop, # remove baselin
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
#         self.predict_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
#                                           nn.ELU(),
#                                           nn.Linear(128,64),
#                                           nn.ELU())
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(64,latent_dim))
#         self.predict_vel_layer = nn.Sequential(nn.Linear(64,3))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         with torch.no_grad():
#             encode = self.predict_encoder(obs_hist[:,-5:,9:].reshape(b,-1)) #remove linvel and command
#             latents = self.predict_latent_layer(encode)
#             predicted_vel = self.predict_vel_layer(encode)

#         actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,l,_ = obs_hist.size()
        
#         # VAE update
#         recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
#         loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
#         # regression
#         with torch.no_grad():
#             _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
#         encode = self.predict_encoder(obs_hist[:,-6:-1,9:].reshape(b,-1))
#         predict_latent = self.predict_latent_layer(encode)
#         predict_vel = self.predict_vel_layer(encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent)
#         mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

#         loss = loss + mseloss + latent_loss
#         return loss

class SlimCausalTransformer(nn.Module):
    def __init__(self, n_embd,n_head,n_layer,dropout,block_size,bias=False) -> None:
        super().__init__()
        # transformer 
        self.transformer = nn.ModuleDict(dict(
            wpe = PositionalEncoding(n_embd,dropout=0.0),
            h = nn.ModuleList([Block(n_embd,n_head,bias,dropout,block_size) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,tok_emb):

        x = self.transformer.wpe(tok_emb.permute(1,0,2))
        x = x.permute(1,0,2)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return x[:,-1,:]
    
class MlpBVAETransRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAETransRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.encoder = StateCausalTransformer(n_obs=num_prop-9,
                                              n_embd=32,
                                              n_head=1,
                                              n_layer=1,
                                              dropout=0.0,
                                              block_size=num_hist-1,
                                              bias=True)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            encode = self.encoder(obs_hist[:,1:,9:])[:,-1,:]
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.encoder(obs_hist[:,:-1,9:])[:,-1,:]
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = loss + mseloss + latent_loss
        return loss

"""
self.encoder = TCN(input_size=in_dim,
                               output_size=32,
                               num_channels=[32,32,32],
                               kernel_size=2)
"""
class MlpBVAETcnRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAETcnRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.cnn_pre_encoder =  nn.Sequential(nn.Linear(num_prop-9,32),
                                          nn.ELU(),
                                          nn.Linear(32,32),
                                          nn.ELU())
        self.encoder = TCN(input_size=32,
                           output_size=32,
                           num_channels=[32,32,32],
                           kernel_size=2)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
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

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            encode = self.encoder(self.cnn_pre_encoder(obs_hist[:,1:,9:]))
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.encoder(self.cnn_pre_encoder(obs_hist[:,:-1,9:]))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

        loss = loss + mseloss + latent_loss
        return loss
    
class MlpBVAETcnContactRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAETcnContactRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop+2, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.cnn_pre_encoder =  nn.Sequential(nn.Linear(num_prop-9,32),
                                          nn.ELU(),
                                          nn.Linear(32,32),
                                          nn.ELU())
        self.encoder = TCN(input_size=32,
                           output_size=32,
                           num_channels=[32,32,32],
                           kernel_size=2)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,2))
        
        self.random = 1
        
        # self.contact_mask_loss = nn.BCEWithLogitsLoss()
        
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
        b,_,_ = obs_hist.size()

        with torch.no_grad():
            encode = self.encoder(self.cnn_pre_encoder(obs_hist[:,1:,9:]))
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,3:]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,contact_mask):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.encoder(self.cnn_pre_encoder(obs_hist[:,:-1,9:]))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_conatct_mask = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_mask_loss = F.mse_loss(predict_conatct_mask,contact_mask)
        
        loss = loss + mseloss + latent_loss + contact_mask_loss
        return loss
    
class MlpBVAETcnContactNoPhaseRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAETcnContactNoPhaseRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=128 + 19 + 4 + 2, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.2,output_dim=num_prop-9) #remove baselin and command
        
        self.encoder = CnnHistoryEncoder(input_size=num_prop-9,
                                         tsteps = 20)
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(128,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(128,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer = nn.Sequential(nn.Linear(128,32),
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
        b,_,_ = obs_hist.size()

        encode = self.encoder(obs_hist[:,1:,9:])
        with torch.no_grad():

            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            
        actor_input = torch.cat([encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,contact_mask):
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        recon,z, mu, log_var = self.Vae(obs_hist[:,:,9:].reshape(b*l,-1)) # remove linvel and command
        loss = self.Vae.loss_fn(obs_hist[:,:,9:].reshape(b*l,-1),recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
            encode = self.encoder(obs_hist[:,:-1,9:])
            
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_conatct_mask = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_mask_loss = F.mse_loss(predict_conatct_mask,contact_mask)
        
        loss = loss + mseloss + latent_loss + contact_mask_loss
        return loss

class PureVqvaeEMA(nn.Module):

    def __init__(self,
                 in_dim= 45,
                 latent_dim = 16,
                 encoder_hidden_dims = [64,32],
                 decoder_hidden_dims = [32,64],
                 output_dim = 45,
                 num_emb=32) -> None:
        
        super(PureVqvaeEMA, self).__init__()

        self.latent_dim = latent_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=num_emb)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        return z,vel

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        z = F.normalize(z)
        return z
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z = self.encode(input)
        quantize,onehot_encode = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,onehot_encode
    
    def loss_fn(self,y, y_hat,quantized,z,onehot_encode):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )
        self.quantizer.update_codebook(z,onehot_encode)

        vq_loss = 0.25*commitment_loss 

        return recon_loss + vq_loss


class InfoNCE(nn.Module):
    """
    对比学习（q: prior侧 / k: teacher侧），含：
    - q/k 投影头（MLP）
    - 负样本队列（memory bank）
    - 温度 tau
    用法：
        loss = nce(q_feats, k_feats)   # q_feats, k_feats: [B, D]
        nce.update_memory(k_feats)     # 每步把 teacher 的 k 放进队列
    """
    def __init__(self, feat_dim=64, proj_dim=128, queue_size=32768, temperature=0.1):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        self.temperature = temperature

        # 负样本队列（注册为buffer，方便保存/加载）
        self.register_buffer("queue", torch.randn(queue_size, proj_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        self.proj_dim = proj_dim

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """把当前 batch 的 keys 放入队列（FIFO），keys: [B, proj_dim]"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[ptr:end, :] = keys
        else:
            first = self.queue_size - ptr
            self.queue[ptr:, :] = keys[:first, :]
            self.queue[:end - self.queue_size, :] = keys[first:, :]

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    @torch.no_grad()
    def update_memory(self, k_feats):
        """将 teacher 的 k 经过投影/归一化后入队（stop-grad 外部已做或这里自动detach）"""
        k = self.k_proj(k_feats.detach())
        k = F.normalize(k, dim=1)
        self._dequeue_and_enqueue(k)

    def forward(self, q_feats, k_feats):
        """
        输入：
            q_feats: prior侧特征 [B, D]
            k_feats: teacher侧特征 [B, D]（会被stop-grad）
        输出：
            InfoNCE loss (标量)
        """
        # 投影 & 归一化
        q = self.q_proj(q_feats)               # [B, P]
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            k = self.k_proj(k_feats.detach())  # stop-gradient
            k = F.normalize(k, dim=1)

        # 正样本相似度：q·k+
        l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(-1)  # [B, 1]

        # 负样本相似度：q·queue^T
        queue = self.queue.clone().detach()                    # [N, P]
        l_neg = torch.einsum("bd,nd->bn", q, queue)            # [B, N]

        # 拼接 logits & 标签
        logits = torch.cat([l_pos, l_neg], dim=1)              # [B, 1+N]
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # 正样本位置=0

        loss = F.cross_entropy(logits, labels)
        return loss

class PriorEncoder(nn.Module):
    def __init__(self, in_dim, feat_dim=16, hidden=(64,32)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers += [nn.Linear(last, feat_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, o):
        q = self.net(o)
        q = F.normalize(q, dim=1)
        return q

class TeacherVqvaeEMA(nn.Module):

    def __init__(self,
                 in_dim= 187,
                 latent_dim = 16,
                 encoder_hidden_dims = [64,32],
                 decoder_hidden_dims = [32,64],
                 output_dim = 45,
                 num_emb=32) -> None:
        
        super(TeacherVqvaeEMA, self).__init__()

        self.latent_dim = latent_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        self.prior_encoder = PriorEncoder(in_dim=80,feat_dim=latent_dim,hidden=(64,32))

       
        self.embedding_dim = latent_dim
        self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=num_emb)

        self.nce = InfoNCE(feat_dim=latent_dim, proj_dim=128, queue_size=32768, temperature=0.1)

    def get_latent(self,input):
        z = self.encode(input)
        return z

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        z = F.normalize(z)
        return z
  
    
    def forward(self, input):
        z = self.encode(input)
        quantize,onehot_encode = self.quantizer(z)
        return quantize,z,onehot_encode
    
    def loss_fn(self, obs_hist, quantized, z, onehot_encode, lambda_div: float = 1e-2):
        q_feats = self.prior_encoder(obs_hist)
        k_feats = quantized.detach()
        loss_nce = self.nce(q_feats, k_feats)
        self.nce.update_memory(k_feats)

        commitment_loss = F.mse_loss(quantized.detach(), z)
        self.quantizer.update_codebook(z, onehot_encode)

        # 使用熵（最大化），作为正则项加入（取负后相加）
        usage = onehot_encode.mean(0) + 1e-12
        entropy = -(usage * usage.log()).sum()
        loss_div = -entropy

        vq_loss = commitment_loss + 0.25 * loss_nce
        return vq_loss + lambda_div * loss_div
# class MlpSimpleLongShortRegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpSimpleLongShortRegressionActor,self).__init__()
        
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims= 16 + 5*latent_dim + 9, # remove baselin
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
        
#         # short long encoder
#         self.single_encoder = nn.Sequential(nn.Linear(num_prop-9,64),
#                                           nn.ELU(),
#                                           nn.Linear(64,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
        
#         self.long_encoder = StateHistoryEncoder(input_size=latent_dim,
#                                                 tsteps=num_hist-1,
#                                                 output_size=16)
#         # estimator 
#         self.estimator_backbone = nn.Sequential(nn.Linear(5*latent_dim+16,64),
#                                           nn.ELU(),
#                                           nn.Linear(64,64),
#                                           nn.ELU())
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
        
#         self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.predict_contact_layer = nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,2))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         latent_history = self.single_encoder(obs_hist)
#         long_latents = self.long_encoder(latent_history) #remove linvel and command
        
#         with torch.no_grad():
#             est_input = torch.cat([latent_history[:,-5:,9:].reshape(b,-1),long_latents])
#             est_latent = self.estimator_backbone(est_input)
#             predicted_latent = self.predict_latent_layer(est_latent)
#             predicted_vel = self.predict_vel_layer(est_latent)
#             predicted_contact = self.predict_contact_layer(est_latent)
            
#         actor_input = torch.cat([latent_history[:,-5:,9:].reshape(b,-1),long_latents,predicted_latent.detach(),predicted_vel.detach(),obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
    
#     def RegLoss(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,l,_ = obs_hist.size()
#         # regression
#         with torch.no_grad():
#             future_latent =  self.single_encoder(obs_hist[:,-1,9:])
            
#         est_latent = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        
#         predict_latent = self.predict_latent_layer(est_latent)
#         predict_vel = self.predict_vel_layer(est_latent)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent.detach())
#         mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())

#         loss = mseloss + latent_loss
#         return loss

# class MlpBVAENoPhaseRegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpBVAENoPhaseRegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + (num_prop-9) + 5 + 4 + 64, # remove baselin
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.Vae = PureVqvaeEMA(in_dim=49,output_dim=49,num_emb=64)#PureBetaVAE(in_dim=49,beta=0.2,output_dim=49) #remove baselin and command
        
#         self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
#                                              tsteps = num_hist-1,
#                                              output_size=32)
        
#         self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
#                                            nn.ELU(),
#                                            nn.Linear(128,64),
#                                            nn.ELU(),
#                                            nn.Linear(64,32),
#                                            nn.ELU())
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
        
#         self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.predict_contact_layer =nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,2))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_encode = self.short_encoder(obs_hist[:,-5:,9:].reshape(b,-1))
#         long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
#         with torch.no_grad():
#             encode = torch.cat([short_encode,long_encode],dim=-1)
#             latents = self.predict_latent_layer(encode)
#             predicted_vel = self.predict_vel_layer(encode)
#             predicted_contact = self.predict_contact_layer(encode)

#         actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,l,_ = obs_hist.size()
        
#         # VAE update
#         vae_input = critic_obs_flatten[:,:49]#torch.cat([height,obs_hist[:,-1,9:]],dim=-1)
#         # recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
#         # loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) # remove linvel and command
#         recon,quantize,z,onehot_encode = self.Vae(vae_input)
#         loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
#         # regression
#         with torch.no_grad():
#             _,_,future_latent,_ = self.Vae(vae_input)
#             short_encode = self.short_encoder(obs_hist[:,-5:,9:].reshape(b,-1))
#             long_encode = self.long_encoder(obs_hist[:,1:,9:])
#             encode = torch.cat([short_encode,long_encode],dim=-1)
            
#         predict_latent = self.predict_latent_layer(encode)
#         predict_vel = self.predict_vel_layer(encode)
#         predict_contact = self.predict_contact_layer(encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent)
#         mseloss = F.mse_loss(predict_vel,obs_hist[:,-1,:3].detach())
#         contact_loss = F.mse_loss(predict_contact,critic_obs_flatten[:,-2:])

#         loss = loss + mseloss + latent_loss + contact_loss
#         return loss

class MlpBVAENoPhaseRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAENoPhaseRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 5 + 4 , # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureVqvaeEMA(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=64)#PureBetaVAE(in_dim=49,beta=0.2,output_dim=49) #remove baselin and command
        
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
        b,_,_ = obs_hist.size()
        
        with torch.no_grad():
            short_encode = self.short_encoder(obs_hist[:,-5:,9:].reshape(b,-1))
            long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)#torch.cat([height,obs_hist[:,-1,9:]],dim=-1)
        # recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        # loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) # remove linvel and command
        recon,quantize,z,onehot_encode = self.Vae(vae_input)
        loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
            short_encode = self.short_encoder(obs_hist[:,-6:-1,9:].reshape(b,-1))
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
            encode = torch.cat([short_encode,long_encode],dim=-1)
            
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_obs_flatten[:,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss

class MlpVQVAERegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAERegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureVqvaeEMA(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=128)#PureBetaVAE(in_dim=49,beta=0.2,output_dim=49) #remove baselin and command
        
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
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
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    

# class MlpVQVAERegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MlpVQVAERegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         self.actor = nn.Sequential(*mlp_factory(activation=activation,
#                                  input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
#                                  out_dims=num_actions,
#                                  hidden_dims=actor_dims))
       
#         self.Vae = PureVqvaeEMA(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=128)#PureBetaVAE(in_dim=49,beta=0.2,output_dim=49) #remove baselin and command
        
#         self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
#                                              tsteps = num_hist-1,
#                                              output_size=16)
        
#         self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
#                                            nn.ELU(),
#                                            nn.Linear(128,64),
#                                            nn.ELU(),
#                                            nn.Linear(64,16),
#                                            nn.ELU())
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
        
#         self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.predict_contact_layer =nn.Sequential(nn.Linear(32,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,2))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist
    
#     def reshape_critic(self,critic_obs_hist_flatten):
#         height = critic_obs_hist_flatten[:,:187]
#         critic_obs_hist = critic_obs_hist_flatten[:,187:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
#         return critic_obs_hist,height

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
#         short_encode = self.short_encoder(short_hist_flatten)
#         long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
#         encode = torch.cat([short_encode,long_encode],dim=-1)
        
#         with torch.no_grad():
#             latents = self.predict_latent_layer(encode)
#             predicted_vel = self.predict_vel_layer(encode)
#             predicted_contact = self.predict_contact_layer(encode)

#         # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
#         actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
#         obs_hist = self.reshape(obs_hist_flatten)
#         critic_hist,height = self.reshape_critic(critic_obs_flatten)
#         b,l,_ = obs_hist.size()
        
#         # VAE update
#         vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
#         recon,quantize,z,onehot_encode = self.Vae(vae_input)
#         loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
#         # regression
#         with torch.no_grad():
#             _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
#             short_hist_flatten = obs_hist[:,-6:-1,9:].reshape(b,-1)
#             short_encode = self.short_encoder(short_hist_flatten)
#             long_encode = self.long_encoder(obs_hist[:,:-1,9:])
#             encode = torch.cat([short_encode,long_encode],dim=-1)
        
#         predict_latent = self.predict_latent_layer(encode)
#         predict_vel = self.predict_vel_layer(encode)
#         predict_contact = self.predict_contact_layer(encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent)
#         mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
#         contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

#         loss = loss + mseloss + latent_loss + contact_loss
#         return loss

class MlpVQVAERegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAERegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 5 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        # self.tanh = nn.Tanh()
       
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        
        # mean = self.tanh(mean)
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
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVAERegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVAERegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,beta=0.02,output_dim=num_prop-9) #remove baselin and command
        
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) # remove linvel and command
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MixmlpVQVAERegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixmlpVQVAERegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        #MixedMlp
        # self.actor = nn.Sequential(*mlp_factory(activation=activation,
        #                          input_dims=latent_dim + (num_prop-9) + 4 + 5 + 32, # remove baselin
        #                          out_dims=num_actions,
        #                          hidden_dims=actor_dims))
        
        self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              latent_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              num_actions=num_actions,
                              hidden_size=64,
                              num_experts=8)
        
        # self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
        #                       latent_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
        #                       num_actions=num_actions,
        #                       hidden_size=64,
        #                       num_experts=8)
       
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input,actor_input)
        # mean = self.actor(actor_input)
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
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MixSlimMlpVQVAERegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixSlimMlpVQVAERegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        #MixedMlp   
        self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              latent_size=32 + 3,
                              num_actions=num_actions,
                              hidden_size=64,
                              num_experts=8)
       
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        latent_input = torch.cat([short_encode,long_encode,obs_hist[:,-1,3:6]],dim=-1)
        mean  = self.actor(latent_input,actor_input)
        # mean = self.actor(actor_input)
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
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MixSlimMlpVaeRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixSlimMlpVaeRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        #MixedMlp   
        self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              latent_size=32 + 3,
                              num_actions=num_actions,
                              hidden_size=64,
                              num_experts=8)
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,output_dim=num_prop-9,beta=0.02)
        
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        latent_input = torch.cat([short_encode,long_encode,obs_hist[:,-1,3:6]],dim=-1)
        mean  = self.actor(latent_input.detach(),actor_input)
        # mean = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) 
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVaeRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVaeRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
  
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,output_dim=num_prop-9,beta=0.02)
        
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)

        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) 
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MixMlpVaeRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixMlpVaeRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        #MixedMlp   
        self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              latent_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              num_actions=num_actions,
                              hidden_size=64,
                              num_experts=8)
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,output_dim=num_prop-9,beta=0.02)
        
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
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input,actor_input)
        # mean = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) 
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MixmlpVqvaeLongShortRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixmlpVqvaeLongShortRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        #MixedMlp
        # self.actor = nn.Sequential(*mlp_factory(activation=activation,
        #                          input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
        #                          out_dims=num_actions,
        #                          hidden_dims=actor_dims))
        
        self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
                              latent_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
                              num_actions=num_actions,
                              hidden_size=64,
                              num_experts=8)
       
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
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(64+16,32),
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
    
    def reshape_critic(self,critic_obs_hist_flatten):
        height = critic_obs_hist_flatten[:,:187]
        critic_obs_hist = critic_obs_hist_flatten[:,187:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
        return critic_obs_hist,height

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:])
        
        with torch.no_grad():
            
            short_est = self.estimator_backbone(short_hist_flatten)
            encode = torch.cat([short_est,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel

        mean  = self.actor(actor_input,actor_input)
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
        
        short_encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        encode = torch.cat([short_encode,long_encode],dim=-1)
        
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss

class MlpVqvaeVelHeightRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeVelHeightRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32 + 16, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureVqvaeEMA(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=128)
        self.height_vae = PureVqvaeEMA(in_dim=121,output_dim=121,num_emb=64)
        
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
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,2))
        
        self.predict_height_layer = nn.Sequential(nn.Linear(64,32),
                                   nn.ELU(),
                                   nn.Linear(32,latent_dim))
        
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
        height = critic_obs_hist_flatten[:,:121]
        critic_obs_hist = critic_obs_hist_flatten[:,121:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
        return critic_obs_hist,height

    def forward(self,obs_hist_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()
        
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
        with torch.no_grad():
            encode = self.estimator_backbone(short_hist_flatten)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            predicted_height = self.predict_height_layer(encode)

        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),predicted_height.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,quantize,z,onehot_encode = self.Vae(vae_input)
        vae_loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
        recon_h,quantize_h,z_h,onehot_encode_h = self.height_vae(height)
        height_vae_loss = self.height_vae.loss_fn(height,recon_h,quantize_h,z_h,onehot_encode_h)
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
            _,_,current_height_latent,_ = self.height_vae(height)
            
        encode = self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1))
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_height = self.predict_height_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])
        height_loss = F.mse_loss(predict_height,current_height_latent)
        
        loss = vae_loss + height_vae_loss + mseloss + latent_loss + contact_loss + height_loss
        return loss
    
# class MixmlpVQVAERegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MixmlpVQVAERegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         #MixedMlp
#         # self.actor = nn.Sequential(*mlp_factory(activation=activation,
#         #                          input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
#         #                          out_dims=num_actions,
#         #                          hidden_dims=actor_dims))
        
#         self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
#                               latent_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
#                               num_actions=num_actions,
#                               hidden_size=64,
#                               num_experts=8)
       
#         self.Vae = PureVqvaeEMA(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=128)
        
#         self.long_encoder = StateHistoryEncoder(input_size=num_prop-8,
#                                              tsteps = num_hist-1,
#                                              output_size=16)
        
#         self.short_encoder = nn.Sequential(nn.Linear((num_prop-8)*5,128),
#                                            nn.ELU(),
#                                            nn.Linear(128,64),
#                                            nn.ELU(),
#                                            nn.Linear(64,16),
#                                            nn.ELU())
        
#         self.estimator_backbone = nn.Sequential(nn.Linear((num_prop-8)*5,128),
#                                           nn.ELU(),
#                                           nn.Linear(128,64),
#                                           nn.ELU())
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
        
#         self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.predict_contact_layer =nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,2))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist
    
#     def reshape_critic(self,critic_obs_hist_flatten):
#         height = critic_obs_hist_flatten[:,:187]
#         critic_obs_hist = critic_obs_hist_flatten[:,187:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
#         return critic_obs_hist,height

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_hist_flatten = obs_hist[:,-5:,8:].reshape(b,-1)
#         short_encode = self.short_encoder(short_hist_flatten)
#         long_encode = self.long_encoder(obs_hist[:,1:,8:]) #remove linvel and command
        
#         with torch.no_grad():
#             encode = self.estimator_backbone(short_hist_flatten)
#             latents = self.predict_latent_layer(encode)
#             predicted_vel = self.predict_vel_layer(encode)
#             predicted_contact = self.predict_contact_layer(encode)

#         actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
#         # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input,actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
#         obs_hist = self.reshape(obs_hist_flatten)
#         critic_hist,height = self.reshape_critic(critic_obs_flatten)
#         b,l,_ = obs_hist.size()
        
#         # VAE update
#         vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
#         recon,quantize,z,onehot_encode = self.Vae(vae_input)
#         loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
#         # regression
#         with torch.no_grad():
#             _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
#         encode = self.estimator_backbone(obs_hist[:,-6:-1,8:].reshape(b,-1))
#         predict_latent = self.predict_latent_layer(encode)
#         predict_vel = self.predict_vel_layer(encode)
#         predict_contact = self.predict_contact_layer(encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent)
#         mseloss = F.mse_loss(predict_vel,0.1*obs_hist[:,-2,:3].detach())
#         contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

#         loss = loss + mseloss + latent_loss + contact_loss
#         return loss

# class MixmlpVQVAERegressionActor(nn.Module):
#     def __init__(self,
#                  num_prop,
#                  num_hist,
#                  actor_dims,
#                  latent_dim,
#                  num_actions,
#                  activation) -> None:
#         super(MixmlpVQVAERegressionActor,self).__init__()
#         self.num_prop = num_prop
#         self.num_hist = num_hist
#         #MixedMlp
#         # self.actor = nn.Sequential(*mlp_factory(activation=activation,
#         #                          input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
#         #                          out_dims=num_actions,
#         #                          hidden_dims=actor_dims))
        
#         self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
#                               latent_size=latent_dim + (num_prop-9) + 4 + 5 + 32,
#                               num_actions=num_actions,
#                               hidden_size=64,
#                               num_experts=8)
       
#         self.Vae = PureVqvaeEMA(in_dim=num_prop-9,output_dim=num_prop-9,num_emb=128)
        
#         self.long_encoder = StateHistoryEncoder(input_size=num_prop-8,
#                                              tsteps = num_hist-1,
#                                              output_size=16)
        
#         self.short_encoder = nn.Sequential(nn.Linear((num_prop-8)*5,128),
#                                            nn.ELU(),
#                                            nn.Linear(128,64),
#                                            nn.ELU(),
#                                            nn.Linear(64,16),
#                                            nn.ELU())
        
#         self.estimator_backbone = nn.Sequential(nn.Linear((num_prop-8)*5,128),
#                                           nn.ELU(),
#                                           nn.Linear(128,64),
#                                           nn.ELU())
        
#         self.predict_latent_layer = nn.Sequential(nn.Linear(64,32),
#                                           nn.ELU(),
#                                           nn.Linear(32,latent_dim))
        
#         self.predict_vel_layer = nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,3))
        
#         self.predict_contact_layer =nn.Sequential(nn.Linear(64,32),
#                                    nn.ELU(),
#                                    nn.Linear(32,2))
        
#         self.random = 1
        
#     def set_random(self,random):
#         self.random = random

#     def reshape(self,obs_hist_flatten):
#         # N*(T*O) -> (N * T)* O -> N * T * O
#         obs_hist_flatten = obs_hist_flatten#.detach()
#         # obs_hist = self.obs_normalizer(obs_hist_flatten.reshape(-1,self.num_prop)).reshape(-1,self.num_hist,self.num_prop)
#         obs_hist = obs_hist_flatten.reshape(-1,self.num_hist,self.num_prop)# add 3 for baselin
#         return obs_hist
    
#     def reshape_critic(self,critic_obs_hist_flatten):
#         height = critic_obs_hist_flatten[:,:187]
#         critic_obs_hist = critic_obs_hist_flatten[:,187:].reshape(-1,5,50 + 19 + 6 + 1)# add 3 for baselin
#         return critic_obs_hist,height

#     def forward(self,obs_hist_flatten):
#         obs_hist = self.reshape(obs_hist_flatten)
#         b,_,_ = obs_hist.size()
        
#         short_hist_flatten = obs_hist[:,-5:,8:].reshape(b,-1)
#         short_encode = self.short_encoder(short_hist_flatten)
#         long_encode = self.long_encoder(obs_hist[:,1:,8:]) #remove linvel and command
        
#         with torch.no_grad():
#             encode = self.estimator_backbone(short_hist_flatten)
#             latents = self.predict_latent_layer(encode)
#             predicted_vel = self.predict_vel_layer(encode)
#             predicted_contact = self.predict_contact_layer(encode)

#         actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
#         # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
#         mean  = self.actor(actor_input,actor_input)
#         return mean
    
#     def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
#         obs_hist = self.reshape(obs_hist_flatten)
#         critic_hist,height = self.reshape_critic(critic_obs_flatten)
#         b,l,_ = obs_hist.size()
        
#         # VAE update
#         vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
#         recon,quantize,z,onehot_encode = self.Vae(vae_input)
#         loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
#         # regression
#         with torch.no_grad():
#             _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
#         encode = self.estimator_backbone(obs_hist[:,-6:-1,8:].reshape(b,-1))
#         predict_latent = self.predict_latent_layer(encode)
#         predict_vel = self.predict_vel_layer(encode)
#         predict_contact = self.predict_contact_layer(encode)
        
#         latent_loss = F.mse_loss(predict_latent,future_latent)
#         mseloss = F.mse_loss(predict_vel,0.1*obs_hist[:,-2,:3].detach())
#         contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

#         loss = loss + mseloss + latent_loss + contact_loss
#         return loss

class MlpVQVAELongShortRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAELongShortRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 5 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
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
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(32,32),
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
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
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
            short_hist_flatten = obs_hist[:,-6:-1,9:].reshape(b,-1)
            short_encode = self.short_encoder(short_hist_flatten)
            long_encode = self.long_encoder(obs_hist[:,:-1,9:]) #remove linvel and command
            encode = torch.cat([short_encode,long_encode],dim=-1)
        
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVqvaeLongEstRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeLongEstRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 5 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        # self.tanh = nn.Tanh()
       
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
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(64+16,32),
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

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],obs_hist[:,-1,8:9]],dim=-1) # remove linvel
        # actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
        
        # mean = self.tanh(mean)
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
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVaeLongShortRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVaeLongShortRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
  
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,output_dim=num_prop-9,beta=0.02)
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                             tsteps = num_hist-1,
                                             output_size=16)
        
        self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,16),
                                           nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(32,32),
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
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)

        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) 
        
        # regression
        with torch.no_grad():
            short_hist_flatten = obs_hist[:,-6:-1,9:].reshape(b,-1)
            short_encode = self.short_encoder(short_hist_flatten)
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = torch.cat([short_encode,long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
    
class MixMlpVaeLongShortRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixMlpVaeLongShortRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
  
        self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              latent_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              num_actions=num_actions,
                              hidden_size=64,
                              num_experts=8)
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,output_dim=num_prop-9,beta=0.02)
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                             tsteps = num_hist-1,
                                             output_size=16)
        
        self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,16),
                                           nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(32,32),
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
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input,actor_input)

        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) 
        
        # regression
        with torch.no_grad():
            short_hist_flatten = obs_hist[:,-6:-1,9:].reshape(b,-1)
            short_encode = self.short_encoder(short_hist_flatten)
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = torch.cat([short_encode,long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MixMlpSlimVaeLongShortRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixMlpSlimVaeLongShortRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
  
        self.actor = MixedMlp(input_size=latent_dim + (num_prop-9) + 4 + 4 + 32,
                              latent_size=32 + 3,
                              num_actions=num_actions,
                              hidden_size=64,
                              num_experts=8)
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,output_dim=num_prop-9,beta=0.02)
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                             tsteps = num_hist-1,
                                             output_size=16)
        
        self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,16),
                                           nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(32,32),
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
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        latent_input = torch.cat([short_encode,long_encode,obs_hist[:,-1,3:6]],dim=-1)
        mean  = self.actor(latent_input,actor_input)

        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) 
        
        # regression
        with torch.no_grad():
            short_hist_flatten = obs_hist[:,-6:-1,9:].reshape(b,-1)
            short_encode = self.short_encoder(short_hist_flatten)
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = torch.cat([short_encode,long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVqvaeLongEstLayerNormRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeLongEstLayerNormRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32,512),
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
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(64+16,32),
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
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
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
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVaeLongShortBothGradRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVaeLongShortBothGradRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
  
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + (num_prop-9) + 4 + 4 + 32, # remove baselin
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = PureBetaVAE(in_dim=num_prop-9,output_dim=num_prop-9,beta=0.02)
        
        self.long_encoder = StateHistoryEncoder(input_size=num_prop-9,
                                             tsteps = num_hist-1,
                                             output_size=16)
        
        self.short_encoder = nn.Sequential(nn.Linear((num_prop-9)*5,128),
                                           nn.ELU(),
                                           nn.Linear(128,64),
                                           nn.ELU(),
                                           nn.Linear(64,16),
                                           nn.ELU())
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(32,32),
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
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)

        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten).detach()
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        critic_hist = critic_hist.detach()
        height = height.detach()
        
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,z, mu, log_var = self.Vae(vae_input) # remove linvel and command
        loss = self.Vae.loss_fn(vae_input,recon,mu,log_var) 
        
        # regression
        with torch.no_grad():
            short_hist_flatten = obs_hist[:,-6:-1,9:].reshape(b,-1)
            
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = torch.cat([short_encode,long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss

class MlpVqvaeLongShortBothGradRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeLongShortBothGradRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
   
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32,512),
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
        
        self.predict_latent_layer = nn.Sequential(nn.Linear(32,32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        self.predict_vel_layer = nn.Sequential(nn.Linear(32,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(32,32),
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
            encode = torch.cat([short_encode,long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)

        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
        mean  = self.actor(actor_input)

        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten).detach()
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        critic_hist = critic_hist.detach()
        height = height.detach()
        
        b,l,_ = obs_hist.size()
        
        # VAE update
        vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        recon,quantize,z,onehot_encode = self.Vae(vae_input)
        loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)
        
        # regression
        with torch.no_grad():
            short_hist_flatten = obs_hist[:,-6:-1,9:].reshape(b,-1)
            
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
        
        encode = torch.cat([short_encode,long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVqvaeLongEstLayerNormFallPredictRegressionTeacherActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeLongEstLayerNormFallPredictRegressionTeacherActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32 + 3 + 187,512),
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

    def forward(self,obs_hist_flatten,critic_obs_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(short_hist_flatten),long_encode],dim=-1)
            latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            predicted_grad_vec = self.predict_gravity_vec_layer(encode)
            
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),predicted_grad_vec.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],height],dim=-1) # remove linvel
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

        loss = loss + mseloss + latent_loss + contact_loss + gravity_loss
        return loss

class MlpVqvaeLongEstLayerNormFallPredictRegressionTeacherVQActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeLongEstLayerNormFallPredictRegressionTeacherVQActor,self).__init__()
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
       
        self.TVae = TeacherVqvaeEMA(in_dim=187,output_dim=num_prop-9,num_emb=16)
        
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

    def forward(self,obs_hist_flatten,critic_obs_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        vq_latent,_,one_hot = self.TVae(height)
        
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(short_hist_flatten),long_encode],dim=-1)
            # latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            predicted_grad_vec = self.predict_gravity_vec_layer(encode)

        prior_latent = self.TVae.prior_encoder(encode)
            
        actor_input = torch.cat([short_encode,long_encode,predicted_vel.detach(),predicted_contact.detach(),predicted_grad_vec.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],vq_latent],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
    
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        # vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        # recon,quantize,z,onehot_encode = self.Vae(vae_input)
        # loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)

       
        
        # regression
        with torch.no_grad():
            _,_,future_latent,_ = self.Vae(obs_hist[:,-1,9:])
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        
        encode = torch.cat([self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1)),long_encode],dim=-1)
        # predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_gra_vec = self.predict_gravity_vec_layer(encode)
        
        # latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:].detach())
        gravity_loss = F.mse_loss(predict_gra_vec,critic_hist[:,-1,6:9].detach())

         # TVAE update
        vq_latent,z,onehot_encode2 = self.TVae(height)
        loss_vq = self.TVae.loss_fn(encode,vq_latent,z,onehot_encode2)

        loss = mseloss  + contact_loss + gravity_loss + loss_vq
        return loss

class MlpVaeLongEstLayerNormFallPredictRegressionTeacherActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVaeLongEstLayerNormFallPredictRegressionTeacherActor,self).__init__()
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

        

        self.Vae = PureTeacherVAE(in_dim=num_prop*num_hist,post_indim=187,latent_dim=16)
        
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

    def forward(self,obs_hist_flatten,critic_obs_flatten):
        obs_hist = self.reshape(obs_hist_flatten)
        b,_,_ = obs_hist.size()

        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        post_latent ,_, _ = self.Vae(obs_hist_flatten,height)
        
        short_hist_flatten = obs_hist[:,-5:,9:].reshape(b,-1)
        short_encode = self.short_encoder(short_hist_flatten)
        long_encode = self.long_encoder(obs_hist[:,1:,9:]) #remove linvel and command
        
        with torch.no_grad():
            encode = torch.cat([self.estimator_backbone(short_hist_flatten),long_encode],dim=-1)
            # latents = self.predict_latent_layer(encode)
            predicted_vel = self.predict_vel_layer(encode)
            predicted_contact = self.predict_contact_layer(encode)
            predicted_grad_vec = self.predict_gravity_vec_layer(encode)

        # prior_latent = self.Vae(obs_hist_flatten,height)
            
        actor_input = torch.cat([short_encode,long_encode,predicted_vel.detach(),predicted_contact.detach(),predicted_grad_vec.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6],post_latent],dim=-1) # remove linvel
        mean  = self.actor(actor_input)
    
        return mean
    
    def VaeLoss(self,obs_hist_flatten,critic_obs_flatten):
        
        obs_hist = self.reshape(obs_hist_flatten)
        critic_hist,height = self.reshape_critic(critic_obs_flatten)
        b,l,_ = obs_hist.size()
        
        # VAE update
        # vae_input = obs_hist[:,:,9:].reshape(b*l,-1)
        # recon,quantize,z,onehot_encode = self.Vae(vae_input)
        # loss = self.Vae.loss_fn(vae_input,recon,quantize,z,onehot_encode)

       
        
        # regression
        with torch.no_grad():
            
            long_encode = self.long_encoder(obs_hist[:,:-1,9:])
        
        encode = torch.cat([self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1)),long_encode],dim=-1)
        
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_gra_vec = self.predict_gravity_vec_layer(encode)
        
       
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:].detach())
        gravity_loss = F.mse_loss(predict_gra_vec,critic_hist[:,-1,6:9].detach())

         # VAE update
        
        loss_vae = self.Vae.loss_fn(obs_hist_flatten,height)

        loss = mseloss  + contact_loss + gravity_loss + loss_vae
        return loss


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

        loss = loss + mseloss + latent_loss + contact_loss + gravity_loss
        return loss
    
class MlpVqvaeLongEstLayerNormCmdScaledRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVqvaeLongEstLayerNormCmdScaledRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32,512),
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
        
        self.predict_contact_layer =nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,2))
        
        self.cmd_scaler = nn.Sequential(nn.Linear(latent_dim + 5 + 32,64),
                                        nn.ELU(),
                                        nn.Linear(64,32),
                                        nn.ELU(),
                                        nn.Linear(32,1),
                                        nn.Sigmoid())
        
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
        
        scaler_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach()],dim=-1)
        cmd_scaled = obs_hist[:,-1,3:6] * self.cmd_scaler(scaler_input)
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),obs_hist[:,-1,9:],cmd_scaled],dim=-1) # remove linvel
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
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:])

        loss = loss + mseloss + latent_loss + contact_loss
        return loss
    
class MlpVqvaeFallAnglePredictRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation
                 ) -> None:
        super(MlpVqvaeFallAnglePredictRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32 + 1,512),
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
        
        self.predict_gravity_simi_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,1))
        # temp
        self.gravity = torch.FloatTensor([0,0,-1]).unsqueeze(0).to('cuda:0')
        
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
            predicted_grad_simi = self.predict_gravity_simi_layer(encode)
            
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),predicted_grad_simi.detach(),obs_hist[:,-1,9:],obs_hist[:,-1,3:6]],dim=-1) # remove linvel
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
            # compute cosine of gravity and global gravity
            next_simi = torch.abs(F.cosine_similarity(self.gravity,critic_hist[:,-1,6:9].detach(),dim=-1))
        
        encode = torch.cat([self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1)),long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_gra_simi = self.predict_gravity_simi_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:].detach())
        gravity_loss = F.mse_loss(predict_gra_simi,next_simi.detach())

        loss = loss + mseloss + latent_loss + contact_loss + gravity_loss
        return loss
    
    
class MlpVqvaeFallAnglePredictScaledCmdRegressionActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation
                 ) -> None:
        super(MlpVqvaeFallAnglePredictScaledCmdRegressionActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        
        self.actor = nn.Sequential(nn.Linear(latent_dim + (num_prop-9) + 4 + 4 + 32 + 1,512),
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
        
        self.predict_gravity_simi_layer = nn.Sequential(nn.Linear(64+16,32),
                                   nn.ELU(),
                                   nn.Linear(32,1))
        
        self.cmd_scale = nn.Sequential(nn.Linear(latent_dim + 5 + 32 + 1 + 3,32),
                                   nn.ELU(),
                                   nn.Linear(32,3))
        # temp
        self.gravity = torch.FloatTensor([0,0,-1]).unsqueeze(0).to('cuda:0')
        
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
            predicted_grad_simi = self.predict_gravity_simi_layer(encode)
        
        cmd_scaled =  self.cmd_scale(torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),predicted_grad_simi.detach(),obs_hist[:,-1,3:6]],dim=-1))
        actor_input = torch.cat([short_encode,long_encode,latents.detach(),predicted_vel.detach(),predicted_contact.detach(),predicted_grad_simi.detach(),obs_hist[:,-1,9:],cmd_scaled],dim=-1) # remove linvel
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
            # compute cosine of gravity and global gravity
            next_simi = torch.abs(F.cosine_similarity(self.gravity,critic_hist[:,-1,6:9].detach(),dim=-1))
        
        encode = torch.cat([self.estimator_backbone(obs_hist[:,-6:-1,9:].reshape(b,-1)),long_encode],dim=-1)
        predict_latent = self.predict_latent_layer(encode)
        predict_vel = self.predict_vel_layer(encode)
        predict_contact = self.predict_contact_layer(encode)
        predict_gra_simi = self.predict_gravity_simi_layer(encode)
        
        latent_loss = F.mse_loss(predict_latent,future_latent)
        mseloss = F.mse_loss(predict_vel,obs_hist[:,-2,:3].detach())
        contact_loss = F.mse_loss(predict_contact,critic_hist[:,-2,-2:].detach())
        gravity_loss = F.mse_loss(predict_gra_simi,next_simi.detach())

        loss = loss + mseloss + latent_loss + contact_loss + gravity_loss
        return loss