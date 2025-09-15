import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import random

import os
import copy

from humanoidGym.learning.algorithms.normalizer import EmpiricalNormalization

from humanoidGym.learning.utils.helper import get_activation, mlp_factory



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

    def _sanitize(self,t, clip=None):
        t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
        return t.clamp(-clip, clip) if clip is not None else t
    
    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output