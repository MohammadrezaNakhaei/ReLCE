from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from typing import Mapping


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class NSequenceBuffer(IterableDataset):
    def __init__(self, max_traj:int=50, seq_len:int=10, N:int=5) -> None:
        super().__init__()
        self.max_traj = max_traj
        self._ptr = 0 # index of list of trajectories
        self._size = 0
        self._trajs = [] # list of dictionaries containing episodes
        self._N = N # number of subtrajectories from one episode
        self._seq_len = seq_len
    
    def __len__(self):
        return self._size

    def add_traj(self, traj: Mapping[str, np.ndarray]):
        if self._size<self.max_traj:
            self._trajs.append(traj)
            self._size += 1
        else:
            self._trajs[self._ptr] = traj
        self._ptr = (self._ptr + 1)%self.max_traj
        
    def _prepare_sample(self, traj_id:int):
        traj = self._trajs[traj_id]
        states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
        len_ep = actions.shape[0]
        idxs = np.random.randint(0, len_ep, size=(self._N))
        seq_states, seq_actions, seq_masks = [],[], [] 
        for idx in idxs:
            state = states[idx: idx+self._seq_len]
            action = actions[idx: idx+self._seq_len]
            mask = np.hstack(
                [np.zeros(state.shape[0]), np.ones(self._seq_len - state.shape[0])]
            )
            seq_states.append(pad_along_axis(state, self._seq_len))
            seq_actions.append(pad_along_axis(action, self._seq_len))
            seq_masks.append(mask)
        seq_states = np.array(seq_states, dtype=np.float32)
        seq_actions = np.array(seq_actions, dtype=np.float32)
        seq_masks = np.array(seq_masks, dtype=np.int8)
        idx = np.random.randint(0, len_ep)
        state = np.array(states[idx], dtype=np.float32)
        next_state = np.array(states[idx + 1], dtype=np.float32)
        action = np.array(actions[idx], dtype=np.float32)
        reward = np.array(rewards[idx], dtype=np.float32)
        done = idx==len_ep-1
        return seq_states, seq_actions, seq_masks, state, action, reward, next_state, done
    
    def __iter__(self):
        while True:
            traj_id = np.random.randint(0, self._size)
            yield self._prepare_sample(traj_id)
