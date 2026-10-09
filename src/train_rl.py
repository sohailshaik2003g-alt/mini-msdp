"""
RL Training Script: SAC with frozen pretrained encoder.
Runs one method × one seed.

Usage:
  python src/train_rl.py --method msdp_r --seed 0 --config configs/default.yaml
  python src/train_rl.py --method concat  --seed 1 --config configs/default.yaml
  python src/train_rl.py --method ablation_pool --seed 0 --config configs/default.yaml
  python src/train_rl.py --method ablation_cls  --seed 0 --config configs/default.yaml
  python src/train_rl.py --method state_sanity  --seed 0 --config configs/default.yaml

State-sanity mode uses state-based SAC (no encoder) to verify env is solvable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import csv
import numpy as np
import torch
from tqdm import tqdm
import yaml

from env import make_env
from model import MSDPEncoder, ConcatEncoder, MSDPPretrainModel, ConcatPretrainModel
from sac import SACAgent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_encoder(method: str, device: str):
    """Load frozen encoder from pretrain checkpoint."""
    if method in ('msdp_r', 'ablation_pool', 'ablation_cls'):
        ckpt_path = 'checkpoints/pretrain_msdp_r.pt'
        ckpt = torch.load(ckpt_path, map_location=device)
        # Reconstruct MSDPPretrainModel and extract encoder
        model = MSDPPretrainModel()
        model.load_state_dict(ckpt['model_state'])
        enc = model.encoder
    elif method == 'concat':
        ckpt_path = 'checkpoints/pretrain_concat.pt'
        ckpt = torch.load(ckpt_path, map_location=device)
        model = ConcatPretrainModel()
        model.load_state_dict(ckpt['model_state'])
        enc = model.encoder
    else:
        return None

    enc = enc.to(device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def evaluate(agent: SACAgent, env, n_episodes: int = 20) -> dict:
    """Run n deterministic episodes, return success rate and mean return."""
    successes = []
    returns   = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)  # deterministic seeds for eval
        ep_return = 0.0
        ep_success = False

        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if info.get('success', False):
                ep_success = True
            if terminated or truncated:
                break

        successes.append(float(ep_success))
        returns.append(ep_return)

    return {
        'success_rate': float(np.mean(successes)),
        'mean_return':  float(np.mean(returns)),
        'std_return':   float(np.std(returns)),
    }


def train(method: str, seed: int, config: dict, device_str: str = None):
    cfg = config
    device = device_str or cfg.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("  [Warning] CUDA not available, using CPU")

    torch.manual_seed(seed)
    np.random.seed(seed)

    total_steps    = cfg['rl']['total_steps']
    eval_every     = cfg['rl']['eval_every']
    eval_episodes  = cfg['rl']['eval_episodes']
    batch_size     = cfg['rl']['batch_size']
    warmup_steps   = cfg['rl']['warmup_steps']
    ckpt_every     = cfg['rl']['checkpoint_every']

    # Environment
    if method == 'state_sanity':
        env  = make_env('state', seed=seed)
        eval_env = make_env('state', seed=seed + 9999)
    else:
        env  = make_env('multisensory', seed=seed)
        eval_env = make_env('multisensory', seed=seed + 9999)

    # Load encoder if needed
    encoder = None
    sac_mode = method
    if method in ('msdp_r', 'concat', 'ablation_pool', 'ablation_cls'):
        encoder = load_encoder(method, device)
        print(f"  Loaded {'MSDP' if 'msdp' in method or 'abl' in method else 'Concat'} encoder")
    elif method == 'state_sanity':
        sac_mode = 'state'

    # Observation spec for replay buffer
    if method == 'state_sanity':
        obs_spec = 9  # flat state dim
    else:
        obs_spec = {'proprio': (14,), 'ft': (4, 6), 'vision': (64, 64, 3)}

    # Build agent
    agent = SACAgent(
        obs_spec=obs_spec,
        action_dim=3,
        device=device,
        mode=sac_mode,
        encoder=encoder,
        d_model=cfg['model']['d_model'],
        lr=cfg['rl']['lr'],
        gamma=cfg['rl']['gamma'],
        tau=cfg['rl']['tau'],
        alpha_init=cfg['rl']['alpha_init'],
        buffer_capacity=cfg['rl']['buffer_capacity'],
        batch_size=batch_size,
    )

    # Logging
    os.makedirs('logs', exist_ok=True)
    log_path = cfg['rl']['log_path'].format(method=method, seed=seed)
    csv_file = open(log_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'success_rate', 'mean_return', 'std_return',
                     'critic_loss', 'actor_loss', 'alpha', 'time_s'])

    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    print(f"\n  RL Training: method={method}, seed={seed}, steps={total_steps}")
    t_start = time.time()
    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_count = 0

    last_metrics = {'critic_loss': 0, 'actor_loss': 0, 'alpha': 0}
    step_times = []

    for step in tqdm(range(1, total_steps + 1), desc=f"RL {method} s={seed}"):
        t_step = time.time()

        # Action selection
        if step <= warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward

        # Store transition (use truncated as done-signal but not for bootstrapping)
        agent.buffer.add(obs, action, reward, next_obs, terminated)

        obs = next_obs
        if terminated or truncated:
            ep_return = 0.0
            ep_count += 1
            obs, _ = env.reset()

        # SAC update
        if step > warmup_steps:
            metrics = agent.update()
            if metrics:
                last_metrics = metrics

        step_times.append(time.time() - t_step)

        # Evaluation
        if step % eval_every == 0:
            eval_metrics = evaluate(agent, eval_env, eval_episodes)
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed
            avg_step_time = np.mean(step_times[-1000:]) * 1000  # ms

            print(f"\n    Step {step:6d}/{total_steps}"
                  f" | SR={eval_metrics['success_rate']:.3f}"
                  f" | Ret={eval_metrics['mean_return']:.2f}"
                  f" | alpha={last_metrics['alpha']:.3f}"
                  f" | {steps_per_sec:.1f} sps"
                  f" | {avg_step_time:.1f}ms/step")

            writer.writerow([
                step,
                eval_metrics['success_rate'],
                eval_metrics['mean_return'],
                eval_metrics['std_return'],
                last_metrics['critic_loss'],
                last_metrics['actor_loss'],
                last_metrics['alpha'],
                elapsed,
            ])
            csv_file.flush()

        # Checkpoint
        if step % ckpt_every == 0:
            ckpt_path = cfg['rl']['save_path'].format(method=method, seed=seed)
            ckpt_path = ckpt_path.replace('_final', f'_step{step}')
            agent.save(ckpt_path)

    # Final save
    final_path = cfg['rl']['save_path'].format(method=method, seed=seed)
    agent.save(final_path)
    csv_file.close()

    total_time = time.time() - t_start
    steps_per_sec = total_steps / total_time
    print(f"\n  Training complete: {total_time:.1f}s ({steps_per_sec:.1f} sps)")
    print(f"  Log: {log_path}")

    return {'method': method, 'seed': seed, 'total_steps': total_steps,
            'time_s': total_time, 'steps_per_sec': steps_per_sec}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['msdp_r', 'concat', 'ablation_pool',
                                             'ablation_cls', 'state_sanity'],
                        default='msdp_r')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--eval-every', type=int, default=None)
    parser.add_argument('--warmup-steps', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.steps is not None:
        config['rl']['total_steps'] = args.steps
    if args.eval_every is not None:
        config['rl']['eval_every'] = args.eval_every
    if args.warmup_steps is not None:
        config['rl']['warmup_steps'] = args.warmup_steps

    train(args.method, args.seed, config, device_str=args.device)
