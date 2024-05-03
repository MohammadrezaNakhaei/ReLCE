import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch

from offline_policy.dynamics import EnsembleDynamics
from utils.termination_fns import get_termination_fn
from utils.logger import Logger, make_log_dirs
from offline_policy.combo import COMBO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument("--uniform-rollout", type=bool, default=True)
    parser.add_argument("--rho-s", type=str, default="model", choices=["model", "mix"])

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--tag', type=str, default='', help='used for logging to distingush different runs')
    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    data = env.get_dataset()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)
    
    # logger
    log_dirs = make_log_dirs(args.task, 'combo', args.seed, vars(args), record_params=['tag'])
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    # dynamic model and training
    dynamics = EnsembleDynamics(
        obs_dim, action_dim, 
        args.dynamics_hidden_dims, 
        args.dynamics_lr, 
        get_termination_fn(args.task),
        num_ensemble=args.n_ensemble, 
        num_elites=args.n_elites, 
        weight_decays=args.dynamics_weight_decay,
        device=args.device)
    dynamics.train(data, logger)
    
    # entropy regularization
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha
    
    combo = COMBO(
        env = env,
        dynamic_module = dynamics, 
        hidden_dims = args.hidden_dims,
        actor_lr = args.actor_lr,
        critic_lr = args.critic_lr,
        tau = args.tau,
        gamma = args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s, 
        device=args.device,
    )
    
    combo.train(
        data,
        epoch = args.epoch,
        step_per_epoch = args.step_per_epoch,
        batch_size = args.batch_size,
        real_ratio = args.real_ratio,
        eval_episodes = args.eval_episodes, 
        rollout_freq = args.rollout_freq,
        rollout_batch_size = args.rollout_batch_size,
        rollout_length = args.rollout_length,
        model_retain_epochs = args.model_retain_epochs ,   
        logger = logger, 
    )

if __name__ == '__main__':
    train()