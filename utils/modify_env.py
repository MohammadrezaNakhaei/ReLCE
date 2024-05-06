import numpy as np
import gym


class TrainingEnv(gym.Env):
    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.mass_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        self.damping_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        model = env.env.wrapped_env.model
        self.original_body_mass = model.body_mass.copy()
        self.original_damping = model.dof_damping.copy()
    
    def reset(self,):
        model = self._env.env.wrapped_env.model
        n_link = model.body_mass.shape[0]
        ind_mass = np.random.randint(len(self.mass_ratios))
        ind_damp = np.random.randint(len(self.damping_ratios))
        for i in range(n_link):
            model.body_mass[i] = self.original_body_mass[i]*self.mass_ratios[ind_mass]
        for i in range(model.dof_damping.shape[0]):
            model.dof_damping[i] = self.original_damping[i]*self.damping_ratios[ind_damp]
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)
    
    def get_normalized_score(self, score):
        return self._env.get_normalized_score(score)
    
    def get_dataset(self,):
        return self._env.get_dataset()