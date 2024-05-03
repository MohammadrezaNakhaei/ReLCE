import numpy as np
import torch 
import torch.nn as nn 
from torch.distributions import Normal

from typing import Dict, List, Union, Tuple, Optional

class TanhNormalWrapper(Normal):
    def log_prob(self, action, raw_action=None):
        if raw_action is None:
            raw_action = self.arctanh(action)
        log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
        eps = 1e-6
        log_prob = log_prob - torch.log((1 - action.pow(2)) + eps).sum(-1, keepdim=True)
        return log_prob

    def mode(self):
        raw_action = self.mean
        action = torch.tanh(self.mean)
        return action, raw_action

    def arctanh(self, x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def rsample(self):
        raw_action = super().rsample()
        action = torch.tanh(raw_action)
        return action, raw_action
    
    
class Actor(nn.Module):
    def __init__(
        self, 
        obs_dim:int, 
        action_dim:int, 
        hidden_dims:Union[List[int], Tuple[int]], 
        activation:nn.Module = nn.ReLU, 
        sigma_min:float = -5.0,
        sigma_max:float = 2.0,
    ) -> None:
        super().__init__()
        neurons = [obs_dim, *hidden_dims, action_dim]
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        layers = nn.ModuleList()
        for l1, l2 in zip(neurons[:-1], neurons[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(activation())
        layers.pop(-1)
        self.mu = nn.Sequential(*layers)
        self.sigma_param = nn.Parameter(torch.zeros(action_dim, 1))
        
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        mu = self.mu(obs)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return TanhNormalWrapper(mu, sigma)
        
        

class Critic(nn.Module):
    def __init__(
        self, 
        obs_dim:int, 
        action_dim:int, 
        hidden_dims:Union[List[int], Tuple[int]], 
        activation:nn.Module = nn.ReLU, 
    ) -> None:
        super().__init__()
        neurons = [obs_dim+action_dim, *hidden_dims, 1]
        layers = nn.ModuleList()
        for l1, l2 in zip(neurons[:-1], neurons[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(activation())
        layers.pop(-1)
        self.q = nn.Sequential(*layers)
        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None, 
    ) -> torch.Tensor:
        if actions is not None:
            obs = torch.cat([obs, actions], dim=1)
        return self.q(obs)