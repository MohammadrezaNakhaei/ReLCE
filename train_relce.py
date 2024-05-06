import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch

from residual_agent.residual_sac import Residual
from offline_policy.combo import COMBO
from offline_policy.dynamics import EnsembleDynamics
from utils.modify_env import TrainingEnv
from utils.logger import load_args, make_log_dirs, Logger
from utils.termination_fns import get_termination_fn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline-path', type=str, required=True, help='Enter the path for offline agent to load')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=int(2e5))
    parser.add_argument('--utd', type=int, default=1, help='number of updates for each interaction')
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)    
    parser.add_argument("--buffer-size", type=int, default=1000, help='number of trajectories in the buffer') 
    parser.add_argument("--latent-dim", type=int, default=8, help='encoder output dim')
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4, help='number of cpu cores used to prepare data')
    parser.add_argument("--encoder-lr", type=float, default=1e-4, help='learning rate for encoder')
    parser.add_argument("--num-same-samples", type=int, default=4, help='number of subtrajectories used to train the encoder, mean value is used for training')
    parser.add_argument("--action-alpha", type=float, default=0.75, help='coefficient for context/offline actions')
    parser.add_argument("--encoder-hidden", type=int, default=16)
    parser.add_argument("--decoder-hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--omega", type=float, default=0.2, help='Coefficient for similarity loss in N points from one trajectory')
    parser.add_argument("--k-steps", type=int, default=5, help='Number of future steps for prediction loss in training the encoder')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", type=str, default='', help='used for logging')
    return parser.parse_args()

def train(args=get_args()):
    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # load offline args
    offline_log_path = os.path.join(args.offline_path, 'record/hyper_param.json')
    offline_args = load_args(offline_log_path)
    
    # logger
    log_dirs = make_log_dirs(offline_args.task, 'relce', args.seed, vars(args), record_params=['tag'])
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    # modify environments
    env = gym.make(offline_args.task)
    eval_env = gym.make(offline_args.task)
    env.seed = args.seed
    eval_env.seed = args.seed
    env = TrainingEnv(env)
    eval_env = TrainingEnv(eval_env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # laod offline agent
    dynamics = EnsembleDynamics(
        obs_dim, action_dim, 
        offline_args.dynamics_hidden_dims, 
        offline_args.dynamics_lr, 
        get_termination_fn(offline_args.task),
        num_ensemble=offline_args.n_ensemble, 
        num_elites=offline_args.n_elites, 
        weight_decays=offline_args.dynamics_weight_decay,
        device=offline_args.device)
    dynamic_path = os.path.join(args.offline_path, 'model')
    dynamics.load(dynamic_path)
    
    if offline_args.auto_alpha:
        target_entropy = offline_args.target_entropy if offline_args.target_entropy \
            else -np.prod(env.action_space.shape)
        offline_args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=offline_args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=offline_args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = offline_args.alpha
        
    combo = COMBO(
        env = env,
        dynamic_module = dynamics, # dynamic not used for evaluation in combo 
        hidden_dims = offline_args.hidden_dims,
        actor_lr = offline_args.actor_lr,
        critic_lr = offline_args.critic_lr,
        tau = offline_args.tau,
        gamma = offline_args.gamma,
        alpha=alpha,
        cql_weight=offline_args.cql_weight,
        temperature=offline_args.temperature,
        max_q_backup=offline_args.max_q_backup,
        deterministic_backup=offline_args.deterministic_backup,
        with_lagrange=offline_args.with_lagrange,
        lagrange_threshold=offline_args.lagrange_threshold,
        cql_alpha_lr=offline_args.cql_alpha_lr,
        num_repeart_actions=offline_args.num_repeat_actions,
        uniform_rollout=offline_args.uniform_rollout,
        rho_s=offline_args.rho_s, 
        device=offline_args.device,
    )
    policy_path = os.path.join(args.offline_path, 'checkpoint/policy.pth')
    combo.load(policy_path)
    
    
    # initializing the residual agent 
    # entropy term
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha
    
    # normalizeing mean and std for decoder prediction objectives
    dataset = env.get_dataset()
    delta = dataset['next_observations']-dataset['observations']
    delta = torch.as_tensor(delta, device=args.device, dtype=torch.float32)
    mu = torch.mean(delta, 0)
    std = torch.std(delta, 0)
    
    res_agent = Residual(
        env, eval_env, combo,
        hidden_dims = args.hidden_dims,
        actor_lr = args.actor_lr, 
        critic_lr = args.critic_lr, 
        tau = args.tau, 
        gamma = args.gamma, 
        alpha = alpha, 
        seq_len = args.seq_len, 
        latent_dim = args.latent_dim, 
        encocer_hidden = args.encoder_hidden,
        decoder_hidden_dims = args.decoder_hidden_dims,
        k_steps = args.k_steps, 
        encoder_lr = args.encoder_lr, 
        omega_consistency = args.omega, 
        buffer_size = args.buffer_size, 
        num_same_samples = args.num_same_samples, 
        batch_size = args.batch_size, 
        num_worker = args.num_workers, 
        action_alpha = args.action_alpha, 
        mu = mu, 
        std = std,
        device=args.device,
    )
 
 
    res_agent.train(
        max_step = args.max_steps, 
        update_ratio =  args.utd, 
        logger = logger, 
        
    )
if __name__ == '__main__':
    train()
    