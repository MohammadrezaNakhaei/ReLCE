from copy import deepcopy
import os, sys
import tqdm, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import gym
from typing import Dict, Union, Tuple, List, Optional
from collections import defaultdict, deque

from offline_policy.combo import COMBO
from utils import Actor, Critic, Logger
from residual_agent.encoder import EncoderModule   
from residual_agent.sequence_buffer import NSequenceBuffer

class Residual:
    def __init__(
        self, 
        env: gym.Env, 
        eval_env: gym.Env,
        offline_agent: COMBO,
        hidden_dims: Union[List[int], Tuple[int]],
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        # encoder parameters
        seq_len:int = 10, 
        latent_dim:int = 16,
        encocer_hidden:int = 16, 
        decoder_hidden_dims:Union[Tuple[int], List[int]]=(256, 256), 
        k_steps:int = 5,
        encoder_lr: float = 0.0001, 
        omega_consistency: float = 0.1,
        mu: torch.Tensor = 0, # normalizing the output
        std: torch.Tensor = 1,         
        # trajectory buffer parameters
        buffer_size:int = 100,
        num_same_samples:int = 4,
        batch_size:int =64,
        num_worker:int = 4,
        # weight of offline vs residual agents 
        action_alpha:float = 0.75,
        device:str = 'cpu',
    ):
        self.env = env 
        self.eval_env = eval_env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.latent_dim = latent_dim 
        # s_res = [obs, action, latent].T
        self.res_obs_dim = self.obs_dim + self.action_dim + self.latent_dim
        
        self.seq_len = seq_len
        self.k_steps = k_steps
        self.num_same_sample = num_same_samples
        self.action_alpha = action_alpha
        self.device = torch.device(device)
    
        self.offline = offline_agent
        assert self.device == self.offline.device, 'offline agent and residual agent should be on the same device'
        # encoder module
        self.encoder = EncoderModule(
            self.obs_dim, self.action_dim, seq_len, latent_dim, 
            encoder_hidden = encocer_hidden, 
            decoder_hidden_dims = decoder_hidden_dims, 
            k_steps = k_steps, 
            learning_rate = encoder_lr, 
            omega_consistency = omega_consistency, 
            mu=mu, std=std,
            device = device
        )
        # residual networks
        self.actor = Actor(self.res_obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic1 = Critic(self.res_obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic2 = Critic(self.res_obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic1_old, self.critic2_old = deepcopy(self.critic1), deepcopy(self.critic2)
        self.actor_optim = Adam(self.actor.parameters(), actor_lr)
        self.critic1_optim =  Adam(self.critic1.parameters(), critic_lr)
        self.critic2_optim = Adam(self.critic2.parameters(), critic_lr) 
        # residual sac params
        self._tau = tau
        self._gamma = gamma
        
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha 
            
        # buffer
        self.buffer = NSequenceBuffer(buffer_size, seq_len+k_steps, num_same_samples)
        self.data_loader = DataLoader(self.buffer, batch_size=batch_size, num_workers=num_worker)
        
        self.coeff = action_alpha
        self.total_t = 0

    def res_actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob
    
    # collect data only using offline policy
    def warm_up(self, n_episodes:int=20):
        for i in range(n_episodes):
            obs, done = self.env.reset(), False
            ep_states = [obs,]
            ep_actions = []
            ep_rewards = []
            while not done:
                action = self.offline.select_action(torch.as_tensor(obs, device=self.device), deterministic=False) # offline policy action
                obs, reward, done, info = self.env.step(action)
                ep_states.append(obs)
                ep_actions.append(action)
                ep_rewards.append(reward)
                self.total_t+=1
            ep_states, ep_actions, ep_rewards = [np.array(lst, dtype=np.float32) for lst in (ep_states, ep_actions, ep_rewards)]
            self.buffer.add_traj(dict(
                states=ep_states, actions=ep_actions, rewards=ep_rewards
            ))
        self.train_iter = iter(self.data_loader) # call next each time for training
        
    @ torch.no_grad()
    def evaluate(self, eval_episodes:int=10, deterministic=True, only_offline=False):
        self._eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        ep_states = [obs,]
        ep_actions = []
        while num_episodes < eval_episodes:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            offline_action, _ = self.offline.actforward(obs_tensor, deterministic=True)
            if only_offline or episode_length<self.seq_len:
                action = offline_action.cpu().numpy()
            else:
                start_seq = -self.seq_len if len(ep_states)<-self.seq_len else -self.seq_len-1 # same length for state sequence 
                context_states = np.array(ep_states[start_seq:-1], dtype=np.float32)
                context_actions = np.array(ep_actions[-self.seq_len:], dtype=np.float32)
                context_state = torch.from_numpy(context_states).unsqueeze(0).to(self.device) # remove last state, add batch dimension
                context_actions = torch.from_numpy(context_actions).unsqueeze(0).to(self.device)
                time_step = torch.arange(context_actions.shape[1]).unsqueeze(0).to(self.device)
                encoded = self.encoder(context_state, context_actions, time_step).squeeze(0) # context encoder
                res_state = self._res_state(obs_tensor, offline_action, encoded) # augmented state
                res_action, _ = self.res_actforward(res_state, deterministic=deterministic) 
                action = self._total_action(offline_action, res_action) 
                action = action.cpu().numpy()
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            ep_states.append(next_obs)
            ep_actions.append(action)
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
                ep_states = [obs]
                ep_actions = []
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        } 
        
    def _res_state(self, obs:torch.Tensor, policy_action:torch.Tensor, encoded:torch.Tensor):
        return torch.cat([obs, policy_action, encoded], dim=-1)
    
    def _total_action(self, offline_action, context_action):
        return self.coeff*offline_action + (1-self.coeff)*context_action
    
    def _res_action_from_total(self, total_action, offline_action):
        return (total_action-self.coeff*offline_action)/(1-self.coeff)
    
    def _prepare_training_samples(self, device:str='cpu'):
        batch = next(self.train_iter)
        seq_states, seq_actions, seq_masks, state, action, reward, next_state, done = [tensor.to(device) for tensor in batch]
        batch_encoder = dict(
            seq_states = seq_states, 
            seq_actions = seq_actions, 
            seq_masks = seq_masks, 
            state = state, 
            action = action, 
            next_state = next_state
            )
        with torch.no_grad():
            offline_act, _ = self.offline.actforward(state, True) # offline policy action
            next_offline_act, _ = self.offline.actforward(next_state, True)
            latents = self.encoder.encode_multiple(seq_states, seq_actions, seq_masks)
            idx = np.random.randint(latents.shape[1])
            latents = latents.mean(1) # batch dim, N, latent dim
        observations = self._res_state(state, offline_act, latents)
        next_observations = self._res_state(next_state, next_offline_act, latents)
        batch_agent = dict(
            observations = observations,
            actions = self._res_action_from_total(action, offline_act), # compute residual actions for training
            next_observations=next_observations,
            rewards = reward,
            terminals = done.to(torch.float32),
        )
        return batch_encoder, batch_agent
    
    def train_sample(self, num_step:int=1):
        self._train()
        for _ in range(num_step):
            batch_encoder, batch_agent = self._prepare_training_samples(self.device)
            loss_encoder = self.encoder.learn_batch(batch_encoder)
            loss_agent = self._learn_res(batch_agent)
        loss_agent.update(loss_encoder)
        return loss_agent
    
    def _learn_res(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.res_actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.res_actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result
    
    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def _eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        
    def save(self, path:str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'encoder': self.encoder.encoder.state_dict(), 
            'decoder': self.encoder.decoder.state_dict(),
        }, path)
        
    def load(self, path:str):
        data = torch.load(path)
        self.actor.load_state_dict(data['actor'])
        self.critic1.load_state_dict(data['critic1'])
        self.critic2.load_state_dict(data['critic2'])    
        self.encoder.encoder.load_state_dict(data['encoder'])
        self.encoder.decoder.load_state_dict(data['decoder'])
        
    def train(self, 
              max_step:int = int(2e5), 
              warm_up:int = 20, 
              update_ratio:int = 1, 
              logger:Logger = None, 
              eval_episodes:int = 10, 
              eval_every:int = 10000,
              save_every: int = 50000,
    ):
        # initial offline evaluation
        eval_info = self.evaluate(only_offline=True)
        
        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        norm_ep_rew_mean = self.env.get_normalized_score(ep_reward_mean) * 100
        norm_ep_rew_std = self.env.get_normalized_score(ep_reward_std) * 100
        if logger:
            logger.log('Offline policy performance:')
            logger.log(f'Mean return: {norm_ep_rew_mean:.3f}, std return: {norm_ep_rew_std:.3f}')
            logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            logger.logkv("eval/episode_length", ep_length_mean)
            logger.logkv("eval/episode_length_std", ep_length_std)
            logger.log('Start training of the residual agent')
            logger.dumpkvs()
        self.warm_up(warm_up)
        best_reward = 0
        while True:
            # start of the episode
            obs, done = self.env.reset(), False
            ep_reward = 0
            ep_length = 0
            # used for saving episode to the buffer
            ep_states = [obs,]
            ep_actions = []
            ep_rewards = []
            while not done and self.total_t<=max_step:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                # action of offline RL agent
                with torch.no_grad():
                    offline_act, _ = self.offline.actforward(obs_tensor, True) # offline policy action
                # first step, no context encoder to be used 
                if len(ep_actions)<self.seq_len:
                    action = offline_act.cpu().numpy()
                else:
                    start_seq = -self.seq_len if len(ep_states)<-self.seq_len else -self.seq_len-1 # same length for state sequence 
                    context_states = np.array(ep_states[start_seq:-1], dtype=np.float32)
                    context_actions = np.array(ep_actions[-self.seq_len:], dtype=np.float32)
                    context_state = torch.from_numpy(context_states).unsqueeze(0).to(self.device) # remove last state, add batch dimension
                    context_actions = torch.from_numpy(context_actions).unsqueeze(0).to(self.device)
                    time_step = torch.arange(context_actions.shape[1]).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        encoded = self.encoder(context_state, context_actions, time_step).squeeze(0) # context encoder, remove batch dim
                        res_state = self._res_state(obs_tensor, offline_act, encoded) # augmented state
                        res_action,_ = self.res_actforward(res_state)
                        action = self._total_action(offline_act, res_action)
                        action = action.cpu().numpy()
                obs, reward, done, info = self.env.step(action)
                ep_states.append(obs)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_reward += reward
                self.total_t+=1
                ep_length += 1

                loss = self.train_sample(update_ratio)
                if logger:
                    for key,val in loss.items():
                        logger.logkv_mean(key, val)
                if self.total_t%eval_every==0:
                    eval_info = self.evaluate(deterministic=True, eval_episodes=eval_episodes)
                    ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                    ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                    norm_ep_rew_mean = self.env.get_normalized_score(ep_reward_mean) * 100
                    norm_ep_rew_std = self.env.get_normalized_score(ep_reward_std) * 100
                    if logger:
                        logger.set_timestep(self.total_t)
                        logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                        logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                        logger.logkv("eval/episode_length", ep_length_mean)
                        logger.logkv("eval/episode_length_std", ep_length_std)
                        logger.dumpkvs()
                        
                    if norm_ep_rew_mean>best_reward and logger:
                        path = os.path.join(logger.model_dir, f'ReLCE_best.pth')
                        self.save(path)
                        best_reward = norm_ep_rew_mean
                if self.total_t%save_every==0 and logger:
                    path = os.path.join(logger.model_dir, f'ReLCE_{self.total_t}.pth')
                    self.save(path)
                    
            if self.total_t>max_step:
                if logger:
                    logger.close()
                break
            
            ep_states, ep_actions, ep_rewards = [np.array(lst, dtype=np.float32) for lst in (ep_states, ep_actions, ep_rewards)]
            self.buffer.add_traj(dict(
                states=ep_states, actions=ep_actions, rewards=ep_rewards
            ))        
            self.train_iter = iter(self.data_loader)   
            ep_reward = self.env.get_normalized_score(ep_reward)*100
            logger.logkv('normalized_episode_reward', ep_reward)      
            logger.logkv('episode_length', ep_length)   
            logger.set_timestep(self.total_t) 
            logger.dumpkvs()  
