from humanoidGym.algo.ppo.normalizer import EmpiricalNormalization
from humanoidGym.algo.ppo.utils import smooth_decay, smooth_decay_se
from .modules import MixMlpSlimVaeLongShortRegressionActor,MlpLongEstLayerNormFallPredictRegressionActor,MlpVqvaeLongEstLayerNormFallPredictRegressionTeacherVQActor, MlpVqvaeLongEstLayerNormFallPredictRegressionTeacherActor ,MixMlpVaeLongShortRegressionActor, MixMlpVaeRegressionActor, MixSlimMlpVQVAERegressionActor, MixmlpVQVAERegressionActor, MixmlpVqvaeLongShortRegressionActor, MlpBVAEDeltaLatentHistRegressionActor, MlpBVAEDeltaRegressionActor, MlpBVAENoPhaseRegressionActor, MlpBVAERegressionActor, MlpBVAETcnContactNoPhaseRegressionActor, MlpBVAETcnContactRegressionActor, MlpBVAETcnRegressionActor, MlpBVAETransRegressionActor, MlpBarlowTwinsLongCnnRegressionActor, MlpBaselineActor, MlpBaselineVQVAEActor, MlpHistoryHeightNoPhaseActor, MlpRnnBVAEActor,MlpSimpleLongShortRegressionActor, MlpSimpleMlpRegressionActor, MlpSimpleRegressionActor, MlpSimpleRnnPhaseShiftRegressionActor, MlpSimpleShortLongRegressionActor, MlpVAERegressionActor, MlpVQVAEActor, MlpVQVAECnnActor, MlpVQVAELongHistActor, MlpVQVAELongShortRegressionActor, MlpVQVAEMixedActor, MlpVQVAERegressionActor, MlpVQVAERnnEncodeActor, MlpVQVAEShortHistActor, MlpVaeLongShortBothGradRegressionActor, MlpVaeLongShortRegressionActor, MlpVaeRegressionActor, MlpVqvaeFallAnglePredictRegressionActor, MlpVqvaeFallAnglePredictScaledCmdRegressionActor, MlpVqvaeLongEstLayerNormCmdScaledRegressionActor, MlpVqvaeLongEstLayerNormFallPredictRegressionActor, MlpVqvaeLongEstLayerNormRegressionActor, MlpVqvaeLongEstRegressionActor, MlpVqvaeLongShortBothGradRegressionActor, MlpVqvaeVelHeightRegressionActor, get_activation,mlp_factory,MlpRnnFullBVAEActor,MlpVQVAERnnActor,MlpRnnBarlowTwinActor,MlpBarlowTwinsRegressionActor,MlpBarlowTwinsRnnRegressionActor,MixedMlpBarlowTwinsRegressionActor,MlpBarlowTwinsCnnRegressionActor, MlpBarlowTwinsCnnRegressionShortHistActor,MlpBarlowTwinsCnnRegressionCurrentActor,MlpBarlowTwinsCnnRegressionShortHistActorNophase,MlpBarlowTwinsCnnRegressionNoPhaseActor,MlpTransRegressionActor, MlpSimSiamActor, MlpBarlowTwinsCnnDeltaRegressionActor,MlpBarlowTwinsCnnSingleActor,MlpBarlowTwinsCnnSingleNoPhaseActor,MlpBarlowTwinsCnnRegressionDirectPastActor,MlpBarlowTwinsCnnRegressionDirectPastNoPhaseActor,MlpSimSiamSingleStepActor,MlpSimSiamSingleStepHeightActor,MlpSimSiamSingleStepNoPhaseActor,MlpBaselineBarlowRegressionActor,MlpBaselineVAEActor,MlpBarlowTwinsNewCnnRegressionNoPhaseActor,MlpBarlowTwinsNewCnnRegressionActor,MlpBaselineTransActor,MlpBaselineTerrianGuideActor,MlpSimpleRnnRegressionActor,MixSlimMlpVaeRegressionActor
from torch.distributions import Normal
import torch.nn as nn
import torch
import torch.optim as optim


class InferenceActor(nn.Module):
    def __init__(self,actor_module,norm_module):
        super().__init__()
        self.actor_module = actor_module
        self.norm_module = norm_module
    def forward(self,x):
        x_norm = self.norm_module(x)
        y = self.actor_module(x_norm)
        return y
    
class InferenceTeacherActor(nn.Module):
    def __init__(self,actor_module,norm_module,critic_norm_module):
        super().__init__()
        self.actor_module = actor_module
        self.norm_module = norm_module
        self.critic_norm_module = critic_norm_module
    
    def forward(self,x):
        x_norm = self.norm_module(x)
        # y_norm = self.critic_norm_module(y)
        # height = y_norm[:,:187]
        a = self.actor_module(x_norm)
        return a

class ActorCritic(nn.Module):
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
        super(ActorCritic, self).__init__()

        self.kwargs = kwargs

        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs

        # self.actor_teacher_backbone = MlpVqvaeLongEstRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        # self.actor_teacher_backbone = MlpVqvaeLongShortBothGradRegressionActor(num_prop=num_prop,#remove linear vel
        #                 num_hist=num_hist,
        #                 num_actions=num_actions,
        #                 actor_dims=[512,256,128],
        #                 activation=activation,
        #                 latent_dim=16)
        #MlpVqvaeLongEstLayerNormRegressionActor MlpVaeLongShortBothGradRegressionActor MlpVqvaeLongShortBothGradRegressionActor
        #MlpVqvaeLongEstLayerNormFallPredictRegressionActor
        #MlpVqvaeLongEstLayerNormCmdScaledRegressionActor
        # self.actor_teacher_backbone = MlpVqvaeLongEstLayerNormCmdScaledRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        # self.actor_teacher_backbone = MlpVqvaeLongEstLayerNormRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        #MlpVqvaeFallAnglePredictRegressionActor
        self.actor_teacher_backbone = MlpLongEstLayerNormFallPredictRegressionActor(num_prop=num_prop,#remove linear vel
                                num_hist=num_hist,
                                num_actions=num_actions,
                                actor_dims=[512,256,128],
                                activation=activation,
                                latent_dim=16)
        # MlpVqvaeFallAnglePredictScaledCmdRegressionActor
        # self.actor_teacher_backbone = MlpVqvaeFallAnglePredictRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        # self.actor_teacher_backbone = MlpVqvaeFallAnglePredictScaledCmdRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        
        # Value function
        critic_layers = mlp_factory(activation,self.num_critic_obs,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        #self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        # self.optimizer = optim.Adam(self.actor_teacher_backbone.Vae.parameters(), lr=3e-4)
        # self.device = next(self.actor_teacher_backbone.parameters()).device
        self.optimizer = optim.Adam(self.actor_teacher_backbone.parameters(), lr=1e-3)
        # self.scaler = torch.amp.GradScaler(self.device)
        # self.optimizer = optim.AdamW(self.actor_teacher_backbone.parameters(), lr=1e-3)
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
        
    def evaluate(self, critic_observations, **kwargs):
        # critic_observations = self.critic_normalize(critic_observations)
        value = self.critic(critic_observations)
        return value
    
    def subtask_loss(self,obs_hist,critic_obs_hist):
        # return self.actor_teacher_backbone.VelLoss(obs_hist)
        # return self.actor_teacher_backbone.BarlowTwinsLoss(obs_hist)
        return self.actor_teacher_backbone.VaeLoss(obs_hist,critic_obs_hist)
        # return self.actor_teacher_backbone.SimSiamLoss(obs_hist)
        #return self.actor_teacher_backbone.NextObsLoss(obs_hist)
        # return self.actor_teacher_backbone.HeightLatentLoss(obs_hist)
        # return self.actor_teacher_backbone.RegLoss(obs_hist,critic_obs_hist)
    
    # def update(self,obs_hist,critic_obs_hist):
    #     loss = self.subtask_loss(obs_hist,critic_obs_hist)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     nn.utils.clip_grad_norm_(self.parameters(),1)
    #     self.optimizer.step()
    #     return loss.detach()
    
    # def update(self,obs_hist,critic_obs_hist):
        
    #     self.optimizer.zero_grad()
    #     with torch.autocast(device_type=str(self.device),dtype=torch.float16):
    #         loss = self.subtask_loss(obs_hist,critic_obs_hist)
            
    #     self.scaler.scale(loss).backward()
    #     self.scaler.unscale_(self.optimizer)
    #     nn.utils.clip_grad_norm_(self.parameters(),1)
    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     self.optimizer.zero_grad()
        
    #     return loss.detach()
    
    def update(self,obs_hist,critic_obs_hist):

        self.optimizer.zero_grad()
        loss = self.subtask_loss(obs_hist,critic_obs_hist)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(),1)
        self.optimizer.step()
        return loss.detach()
    
    def set_random(self,it):
        random = smooth_decay_se(it,3000,2000,1,0.2)    
        self.actor_teacher_backbone.set_random(random)


class ActorCritic_Teacher(nn.Module):
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
        super(ActorCritic_Teacher, self).__init__()

        self.kwargs = kwargs

        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs

        # self.actor_teacher_backbone = MlpVqvaeLongEstRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        # self.actor_teacher_backbone = MlpVqvaeLongShortBothGradRegressionActor(num_prop=num_prop,#remove linear vel
        #                 num_hist=num_hist,
        #                 num_actions=num_actions,
        #                 actor_dims=[512,256,128],
        #                 activation=activation,
        #                 latent_dim=16)
        #MlpVqvaeLongEstLayerNormRegressionActor MlpVaeLongShortBothGradRegressionActor MlpVqvaeLongShortBothGradRegressionActor
        #MlpVqvaeLongEstLayerNormFallPredictRegressionActor
        #MlpVqvaeLongEstLayerNormCmdScaledRegressionActor
        # self.actor_teacher_backbone = MlpVqvaeLongEstLayerNormCmdScaledRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        # self.actor_teacher_backbone = MlpVqvaeLongEstLayerNormRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        #MlpVqvaeFallAnglePredictRegressionActor
        self.actor_teacher_backbone = MlpLongEstLayerNormFallPredictRegressionActor(num_prop=num_prop,#remove linear vel
                                num_hist=num_hist,
                                num_actions=num_actions,
                                actor_dims=[512,256,128],
                                activation=activation,
                                latent_dim=16)
        # MlpVqvaeFallAnglePredictScaledCmdRegressionActor
        # self.actor_teacher_backbone = MlpVqvaeFallAnglePredictRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        # self.actor_teacher_backbone = MlpVqvaeFallAnglePredictScaledCmdRegressionActor(num_prop=num_prop,#remove linear vel
        #                         num_hist=num_hist,
        #                         num_actions=num_actions,
        #                         actor_dims=[512,256,128],
        #                         activation=activation,
        #                         latent_dim=16)
        
        # Value function
        critic_layers = mlp_factory(activation,self.num_critic_obs,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        #self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        # self.optimizer = optim.Adam(self.actor_teacher_backbone.Vae.parameters(), lr=3e-4)
        # self.device = next(self.actor_teacher_backbone.parameters()).device
        self.optimizer = optim.Adam(self.actor_teacher_backbone.parameters(), lr=1e-3)
        # self.scaler = torch.amp.GradScaler(self.device)
        # self.optimizer = optim.AdamW(self.actor_teacher_backbone.parameters(), lr=1e-3)
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

    def update_distribution(self, obs,critic_obs):
        mean = self.act_inference(obs,critic_obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,critic_obs,**kwargs):
        self.update_distribution(obs,critic_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self,obs_hist,critic_obs, **kwargs):
        mean = self.actor_teacher_backbone(obs_hist,critic_obs)
        return mean
        
    def evaluate(self, critic_observations, **kwargs):
        # critic_observations = self.critic_normalize(critic_observations)
        value = self.critic(critic_observations)
        return value
    
    def subtask_loss(self,obs_hist,critic_obs_hist):
        # return self.actor_teacher_backbone.VelLoss(obs_hist)
        # return self.actor_teacher_backbone.BarlowTwinsLoss(obs_hist)
        return self.actor_teacher_backbone.VaeLoss(obs_hist,critic_obs_hist)
        # return self.actor_teacher_backbone.SimSiamLoss(obs_hist)
        #return self.actor_teacher_backbone.NextObsLoss(obs_hist)
        # return self.actor_teacher_backbone.HeightLatentLoss(obs_hist)
        # return self.actor_teacher_backbone.RegLoss(obs_hist,critic_obs_hist)
    
    # def update(self,obs_hist,critic_obs_hist):
    #     loss = self.subtask_loss(obs_hist,critic_obs_hist)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     nn.utils.clip_grad_norm_(self.parameters(),1)
    #     self.optimizer.step()
    #     return loss.detach()
    
    # def update(self,obs_hist,critic_obs_hist):
        
    #     self.optimizer.zero_grad()
    #     with torch.autocast(device_type=str(self.device),dtype=torch.float16):
    #         loss = self.subtask_loss(obs_hist,critic_obs_hist)
            
    #     self.scaler.scale(loss).backward()
    #     self.scaler.unscale_(self.optimizer)
    #     nn.utils.clip_grad_norm_(self.parameters(),1)
    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     self.optimizer.zero_grad()
        
    #     return loss.detach()
    
    def update(self,obs_hist,critic_obs_hist):

        self.optimizer.zero_grad()
        loss = self.subtask_loss(obs_hist,critic_obs_hist)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(),1)
        self.optimizer.step()
        return loss.detach()
    
    def set_random(self,it):
        random = smooth_decay_se(it,3000,2000,1,0.2)    
        self.actor_teacher_backbone.set_random(random)
        
    
