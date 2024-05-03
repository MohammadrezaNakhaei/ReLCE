from copy import deepcopy
import os, sys
import tqdm, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import gym
from typing import Dict, Union, Tuple, List, Optional
from collections import defaultdict, deque

from offline_policy.dynamics import EnsembleDynamics
from offline_policy.buffer import ReplayBuffer
from utils import Actor, Critic, Logger


class COMBO:
    def __init__(
        self, 
        env: gym.Env, 
        dynamic_module: EnsembleDynamics,
        hidden_dims: Union[List[int], Tuple[int]],
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        uniform_rollout: bool = False,
        rho_s: str = "mix",
        device:str = 'cpu',
    ):
        self.env = env
        self.dynamics = dynamic_module
        self.device = torch.device(device)
        assert self.dynamics.device == self.device, 'dynamic model and policy should be on the same device'
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.hidden_dims = hidden_dims
        
        self.actor = Actor(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.actor_old = deepcopy(self.actor)
        self.critic1 = Critic(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic1_old, self.critic2_old = deepcopy(self.critic1) , deepcopy(self.critic2)
        self.actor_optim = Adam(self.actor.parameters(), actor_lr)
        self.critic1_optim =  Adam(self.critic1.parameters(), critic_lr)
        self.critic2_optim = Adam(self.critic2.parameters(), critic_lr)     
        
        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha        

        self._cql_weight = cql_weight
        self._temperature = temperature
        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._with_lagrange = with_lagrange
        self._lagrange_threshold = lagrange_threshold

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self._num_repeat_actions = num_repeart_actions     
           
        self._uniform_rollout = uniform_rollout
        self._rho_s = rho_s

    def actforward(
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

    @ torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()
    
    def train(
        self, 
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        rollout_freq:int = 1000, 
        rollout_batch_size:int = 50_000,
        rollout_length:int = 5,
        model_retain_epochs:int = 5,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Logger=None,
    ):
        device = self.device
        data = self.env.get_dataset()
        data_size = data['observations'].shape[0]
        real_buffer = ReplayBuffer(data_size, self.obs_dim, self.action_dim,device)
        real_buffer.load_dataset(data)
        fake_buffer = ReplayBuffer(rollout_batch_size*rollout_length*model_retain_epochs, 
                                   self.obs_dim, self.action_dim, device)
        
        start_time = time.time()
        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        for e in range(1, epoch + 1):
            self.train()
            pbar = tqdm(range(step_per_epoch), desc=f"Epoch #{e}/{epoch}")
            for it in pbar:
                if num_timesteps % rollout_freq == 0:
                    init_obss = real_buffer.sample(rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.rollout(init_obss, rollout_length)
                    fake_buffer.add_batch(**rollout_transitions)
                    if logger:
                        logger.log(
                            "num rollout transitions: {}, reward mean: {:.4f}".\
                                format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                        )
                        for _key, _value in rollout_info.items():
                            logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(batch_size * real_ratio)
                fake_sample_size = batch_size - real_sample_size
                real_batch = real_buffer.sample(batch_size=real_sample_size)
                fake_batch = fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.learn(batch)
                pbar.set_postfix(**loss)
                if logger:
                    for k, v in loss.items():
                        logger.logkv_mean(k, v)
                
                num_timesteps += 1
            if lr_scheduler is not None:
                lr_scheduler.step()
            # evaluate current policy
            eval_info = self._evaluate(eval_episodes)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            if logger:
                logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                logger.logkv("eval/episode_length", ep_length_mean)
                logger.logkv("eval/episode_length_std", ep_length_std)
                logger.set_timestep(num_timesteps)
                logger.dumpkvs(exclude=["dynamics_training_progress"])
                self.save(os.path.join(logger.checkpoint_dir, "policy.pth"))

        if logger:
            logger.log("total time: {:.2f}s".format(time.time() - start_time))
            self.save(os.path.join(logger.model_dir, "policy.pth"))
            logger.close()
        return {"last_10_performance": np.mean(last_10_performance)}
    
    def _evaluate(self, eval_episodes:int=10):
        self.eval()
        obs = self.env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        while num_episodes < eval_episodes:
            action = select_action(obs.reshape(1, -1), deterministic=True)
            next_obs, reward, terminal, _ = self.env.step(action.flatten())
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
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }      
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]
        
        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = torch.min(
                    self.critic1_old(next_obss, next_actions),
                    self.critic2_old(next_obss, next_actions)
                )
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        if self._rho_s == "model":
            obss, actions, next_obss = fake_batch["observations"], \
                fake_batch["actions"], fake_batch["next_observations"]
            
        batch_size = len(obss)
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
        # Samples from the original dataset
        real_obss, real_actions = real_batch['observations'], real_batch['actions']
        q1, q2 = self.critic1(real_obss, real_actions), self.critic2(real_obss, real_actions)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/conservative1": conservative_loss1.item()/self._cql_weight,
            "loss/conservative2": conservative_loss1.item()/self._cql_weight,
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        return result

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            if self._uniform_rollout:
                actions = np.random.uniform(
                    self.action_space.low[0],
                    self.action_space.high[0],
                    size=(len(observations), self.action_space.shape[0])
                )
            else:
                actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
    

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi)

        q1 = self.critic1(obs_to_pred, act)
        q2 = self.critic2(obs_to_pred, act)

        return q1 - log_prob.detach(), q2 - log_prob.detach()

    def calc_random_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.critic1(obs, random_act)
        q2 = self.critic2(obs, random_act)

        log_prob1 = np.log(0.5**random_act.shape[-1])
        log_prob2 = np.log(0.5**random_act.shape[-1])

        return q1 - log_prob1, q2 - log_prob2

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        
    def save(self, path:str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, path)
        
    def load(self, path:str):
        data = torch.load(path)
        self.actor.load_state_dict(data['actor'])
        self.critic1.load_state_dict(data['critic1'])
        self.critic2.load_state_dict(data['critic2'])