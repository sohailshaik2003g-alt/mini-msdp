"""
Pretraining script: MSDP-R and Concat baseline.

Usage:
  python src/pretrain.py --method msdp_r --config configs/default.yaml
  python src/pretrain.py --method concat  --config configs/default.yaml

Outputs:
  - checkpoints/pretrain_{method}.pt    (encoder weights)
  - results/figures/recon_{method}.png  (reconstruction figure)
  - logs/pretrain_{method}.csv          (loss curves)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yaml
from env import make_env
from model import MSDPPretrainModel, ConcatPretrainModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_transitions(env, n: int, seed: int = 0) -> list[dict]:
    """Collect n random-policy transitions from env."""
    print(f"  Collecting {n} random transitions...")
    transitions = []
    obs, _ = env.reset(seed=seed)
    for i in tqdm(range(n), desc="Collecting data"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append({
            'obs':      {k: v.copy() for k, v in obs.items()},
            'action':   action.copy(),
            'next_obs': {k: v.copy() for k, v in next_obs.items()},
            'reward':   reward,
            'done':     terminated or truncated,
        })
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    print(f"  Done. Collected {len(transitions)} transitions.")
    return transitions


def make_batch(transitions: list, indices: np.ndarray, device: str) -> dict:
    """Build batch from list of transitions."""
    batch = transitions[0]['obs']  # just for keys
    obs_keys = list(batch.keys())

    obs = {k: torch.stack([
        torch.as_tensor(transitions[i]['obs'][k], dtype=torch.float32)
        for i in indices
    ]).to(device) for k in obs_keys}

    return {'obs': obs}


def save_reconstruction_figure(model, batch_obs, device, path, method):
    """Visualise reconstruction quality."""
    model.eval()
    with torch.no_grad():
        if method == 'msdp_r':
            result = model(batch_obs)
            pred_vision = result['pred']['vision']   # (B, 64, 3)
            tgt_vision  = result['target']['vision'] # (B, 64, 3)

            # Reshape to grid: 8×8 patches → 64×64 image
            B = pred_vision.shape[0]
            pred_img = pred_vision[:4].reshape(4, 8, 8, 3).clamp(0, 1)
            tgt_img  = tgt_vision[:4].reshape(4, 8, 8, 3).clamp(0, 1)

            fig, axes = plt.subplots(2, 4, figsize=(10, 5))
            fig.suptitle(f'MSDP-R Reconstruction (patch-mean targets), mask={0.7:.0%}')
            for i in range(4):
                axes[0, i].imshow(tgt_img[i].cpu().numpy())
                axes[0, i].set_title('Target')
                axes[0, i].axis('off')
                axes[1, i].imshow(pred_img[i].cpu().numpy())
                axes[1, i].set_title('Pred')
                axes[1, i].axis('off')

        elif method == 'concat':
            result = model(batch_obs)
            # Show loss info only
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"Concat loss: {result['loss'].item():.4f}",
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Concat Baseline Pretraining')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved reconstruction figure: {path}")
    model.train()


def pretrain(method: str, config: dict, device_str: str = None):
    cfg = config
    device = device_str or cfg.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("  [Warning] CUDA not available, using CPU")

    # Environment
    env = make_env('multisensory', seed=0)

    # Collect data
    t0 = time.time()
    n_transitions = cfg['pretrain']['n_transitions']
    transitions = collect_transitions(env, n_transitions, seed=0)
    collect_time = time.time() - t0
    print(f"  Data collection time: {collect_time:.1f}s ({n_transitions/collect_time:.1f} steps/s)")

    # Build model
    if method == 'msdp_r':
        model = MSDPPretrainModel(
            mask_ratio=cfg['pretrain']['mask_ratio']
        ).to(device)
    elif method == 'concat':
        model = ConcatPretrainModel().to(device)
    else:
        raise ValueError(f"Unknown method: {method}")

    lr = cfg['pretrain']['lr']
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = cfg['pretrain']['batch_size']
    n_steps = cfg['pretrain']['n_steps']

    # Logging
    os.makedirs('logs', exist_ok=True)
    log_path = f"logs/pretrain_{method}.csv"
    csv_file = open(log_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'loss_proprio', 'loss_ft', 'loss_vision', 'time_s'])

    # Measure throughput
    t_start = time.time()
    steps_sec = 0.0

    print(f"\n  Pretraining {method} for {n_steps} steps...")
    model.train()
    idx_array = np.arange(n_transitions)

    for step in tqdm(range(n_steps), desc=f"Pretrain {method}"):
        indices = np.random.choice(idx_array, size=batch_size, replace=True)
        batch = make_batch(transitions, indices, device)
        obs = batch['obs']

        result = model(obs)
        loss = result['loss']

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0:
            elapsed = time.time() - t_start
            steps_sec = (step + 1) / max(elapsed, 1e-6)
            lp = result.get('loss_proprio', torch.tensor(0.0)).item()
            lf = result.get('loss_ft',      torch.tensor(0.0)).item()
            lv = result.get('loss_vision',  torch.tensor(0.0)).item()
            writer.writerow([step, loss.item(), lp, lf, lv, elapsed])
            if step % 5000 == 0:
                print(f"    Step {step:5d} | loss={loss.item():.4f} "
                      f"(p={lp:.4f} ft={lf:.4f} v={lv:.4f}) | {steps_sec:.1f} steps/s")

    csv_file.close()
    total_time = time.time() - t_start
    print(f"  Pretraining done in {total_time:.1f}s ({steps_sec:.1f} steps/s)")

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = f"checkpoints/pretrain_{method}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'method': method,
        'steps': n_steps,
        'final_loss': loss.item(),
    }, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    # Save reconstruction figure (using first 4 samples)
    sample_indices = np.arange(min(4, n_transitions))
    sample_batch = make_batch(transitions, sample_indices, device)
    fig_path = cfg['pretrain']['recon_figure_path'].format(method=method)
    save_reconstruction_figure(model, sample_batch['obs'], device, fig_path, method)

    return model, transitions, total_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['msdp_r', 'concat'], default='msdp_r')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    pretrain(args.method, config, device_str=args.device)
