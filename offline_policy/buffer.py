### code adopted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/buffer/buffer.py
import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((self._max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=np.float32)
        next_observations = np.array(dataset["next_observations"], dtype=np.float32)
        actions = np.array(dataset["actions"], dtype=np.float32)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations": torch.from_numpy(self.observations[batch_indexes]).to(self.device),
            "actions": torch.from_numpy(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.from_numpy(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.from_numpy(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[batch_indexes]).to(self.device)
        }
if __name__ == '__main__':
    import d4rl, gym
    env = gym.make('hopper-medium-replay-v2')
    data = env.get_dataset() 
    buffer = ReplayBuffer(int(1e6), env.observation_space.shape[0], env.action_space.shape[0])
    buffer.load_dataset(data)
    samples = buffer.sample(64)
    print(samples.keys())
    print(samples['observations'].shape)
    