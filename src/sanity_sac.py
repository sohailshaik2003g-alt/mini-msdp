"""
State-based SAC sanity check: verify the mock environment is learnable.
Uses flat state observations (9-dim), no encoder.
Run: python src/sanity_sac.py

Expected: SR > 0.5 within 20k steps on mock env (very solvable).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from env import make_env
from sac import SACAgent


def run_sanity(total_steps: int = 30000, seed: int = 0, device: str = 'cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env('state', seed=seed)
    eval_env = make_env('state', seed=seed + 9999)

    agent = SACAgent(
        obs_spec=9,
        action_dim=3,
        device=device,
        mode='state',
        buffer_capacity=50000,
        batch_size=256,
        lr=3e-4,
    )

    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/sanity_sac_seed{seed}.csv'
    csv_file = open(log_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'success_rate', 'mean_return'])

    obs, _ = env.reset(seed=seed)
    eval_steps, eval_sr = [], []
    t_start = time.time()

    for step in tqdm(range(1, total_steps + 1), desc=f"State SAC seed={seed}"):
        if step <= 1000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()

        if step > 1000:
            agent.update()

        if step % 2500 == 0:
            # Eval
            successes, returns = [], []
            for ep in range(20):
                o, _ = eval_env.reset(seed=ep + 1000)
                ep_r, done_flag = 0.0, False
                success = False
                while not done_flag:
                    a = agent.select_action(o, deterministic=True)
                    o, r, te, tr, info = eval_env.step(a)
                    ep_r += r
                    if info.get('success', False):
                        success = True
                    done_flag = te or tr
                successes.append(float(success))
                returns.append(ep_r)

            sr = np.mean(successes)
            mr = np.mean(returns)
            eval_steps.append(step)
            eval_sr.append(sr)
            elapsed = time.time() - t_start
            print(f"  Step {step:5d} | SR={sr:.3f} | Ret={mr:.2f} | {step/elapsed:.0f} sps")
            writer.writerow([step, sr, mr])
            csv_file.flush()

    csv_file.close()

    # Plot
    os.makedirs('results/figures', exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(eval_steps, eval_sr, 'b-o', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.title(f'State-based SAC Sanity Check (seed={seed})')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/sanity_sac.png', dpi=100)
    plt.close()
    print(f"  Sanity check done. Final SR: {eval_sr[-1]:.3f}")
    print(f"  Log: {log_path}")
    print(f"  Plot: results/figures/sanity_sac.png")
    return eval_sr


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run_sanity(total_steps=args.steps, seed=args.seed, device=args.device)
