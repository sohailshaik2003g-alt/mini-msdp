"""
Attention map visualization script for MSDP Critic cross-attention.
Visualizes how the learned critic query attends over:
  - Token 0: Proprioception (R^14)
  - Token 1: Force-Torque (R^24)
  - Tokens 2..65: 8x8 Vision patches (RGB 64x64)

Usage:
  python src/visualize_attention.py --checkpoint checkpoints/pretrain_msdp_r.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env import make_env
from model import MSDPEncoder
from sac import CrossAttentionCritic


def extract_and_plot_attention(
    encoder_path: str = "checkpoints/pretrain_msdp_r.pt",
    rl_critic_path: str = None,
    output_path: str = "results/figures/attention_maps.png",
    device: str = "cpu",
    seed: int = 42,
):
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load encoder
    encoder = MSDPEncoder(d_model=128).to(device)
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location=device)
        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        # Filter encoder weights if whole pretrain model was saved
        enc_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        if enc_state:
            encoder.load_state_dict(enc_state)
        else:
            encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded encoder from {encoder_path}")
    else:
        print(f"Warning: {encoder_path} not found, using initialized weights.")
    encoder.eval()

    # 2. Setup Critic cross-attention
    critic = CrossAttentionCritic(d_model=128, action_dim=3, n_heads=4).to(device)
    if rl_critic_path and os.path.exists(rl_critic_path):
        c_ckpt = torch.load(rl_critic_path, map_location=device)
        critic.load_state_dict(c_ckpt.get('critic_state', c_ckpt), strict=False)
        print(f"Loaded critic from {rl_critic_path}")
    critic.eval()

    # 3. Create env and capture several test states (e.g. initial vs near contact)
    env = make_env('multisensory', seed=seed)
    
    samples = []
    obs, _ = env.reset(seed=seed)
    for step in range(30):
        # Apply action pushing towards cube
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        next_obs, r, te, tr, info = env.step(action)
        # Record a non-contact sample early, and a contact sample later
        if step == 2:
            samples.append(('Pre-contact Step 2', obs))
        elif step == 15:
            samples.append(('In-contact Step 15', obs))
        obs = next_obs
        if len(samples) >= 2:
            break
    if len(samples) < 2:
        samples.append(('End Step', obs))

    # 4. Compute attention for each sample
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(len(samples), 3, figsize=(14, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, 0)

    for i, (label, sample_obs) in enumerate(samples):
        # Prep tensors
        t_obs = {
            'proprio': torch.from_numpy(sample_obs['proprio']).unsqueeze(0).to(device),
            'ft': torch.from_numpy(sample_obs['ft']).unsqueeze(0).to(device),
            'vision': torch.from_numpy(sample_obs['vision']).unsqueeze(0).to(device),
        }

        with torch.no_grad():
            embeddings = encoder(t_obs)  # (1, 66, 128)
            q = critic.query.expand(1, -1, -1)  # (1, 1, 128)
            # Compute cross-attention weights: (1, 1, 66)
            _, attn_weights = critic.cross_attn(q, embeddings, embeddings, need_weights=True)
            attn = attn_weights.squeeze(0).squeeze(0).cpu().numpy()  # (66,)

        proprio_attn = attn[0]
        ft_attn = attn[1]
        vision_attn_grid = attn[2:66].reshape(8, 8)

        # Plot 1: Observation RGB image
        axes[i, 0].imshow(sample_obs['vision'])
        axes[i, 0].set_title(f"{label}: Observation (64x64 RGB)")
        axes[i, 0].axis('off')

        # Plot 2: 8x8 Vision Attention Heatmap
        im = axes[i, 1].imshow(vision_attn_grid, cmap='viridis', interpolation='nearest')
        axes[i, 1].set_title(f"Vision Attention (8x8 Grid)\nTotal Vis Weight: {vision_attn_grid.sum():.3f}")
        axes[i, 1].set_xlabel("Patch X")
        axes[i, 1].set_ylabel("Patch Y")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Plot 3: Modality Weight Distribution
        modalities = ['Proprio (1)', 'Force-Torque (1)', 'Vision (64 tokens)']
        weights = [proprio_attn, ft_attn, vision_attn_grid.sum()]
        bars = axes[i, 2].bar(modalities, weights, color=['#3b82f6', '#ef4444', '#10b981'])
        axes[i, 2].set_title(f"Cross-Attention Modality Allocation")
        axes[i, 2].set_ylabel("Attention Weight")
        axes[i, 2].set_ylim(0, 1.05)
        for bar, w in zip(bars, weights):
            axes[i, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f"{w:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Attention visualization saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/pretrain_msdp_r.pt')
    parser.add_argument('--critic', default=None)
    parser.add_argument('--output', default='results/figures/attention_maps.png')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    extract_and_plot_attention(args.checkpoint, args.critic, args.output, args.device)
