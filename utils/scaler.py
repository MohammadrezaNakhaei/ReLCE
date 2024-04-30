## code adopted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/utils/scaler.py
import numpy as np
import os.path as path
import torch


class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, data):
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        return self.std * data + self.mu
    
    def save_scaler(self, save_path):
        mu_path = path.join(save_path, "mu.npy")
        std_path = path.join(save_path, "std.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path):
        mu_path = path.join(load_path, "mu.npy")
        std_path = path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data