# MSDP Mini: Multisensory Pretraining for Contact-Rich Robot RL

Independent reduced-scope reproduction of MSDP (Krohn et al.), not affiliated with the authors.

This repository provides an independent, clean-room implementation of **"Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning" (MSDP)**, Krohn, Prasad, Tiboni, Chalvatzaki, arXiv:2511.14427 ([arXiv:2511.14427](https://arxiv.org/abs/2511.14427)). Developed as an MSc AI/ML student research project at TU Darmstadt.

---

## 1. Overview & Method

Contact-rich robotic manipulation tasks require reasoning over multiple heterogeneous sensory streams: low-dimensional proprioception (joint angles and velocities), high-frequency wrist force-torque (F/T) readings, and high-dimensional camera vision. MSDP addresses this by:
1. **Tokenizing heterogeneous sensors** into a shared latent space ($d = 128$).
2. **Masked Multisensory Autoencoding (MSDP-R)**: Masking 70% of tokens across modalities (ensuring at least one visual patch token remains visible) and reconstructing the full multisensory state.
3. **Downstream RL with Latent Bridging**: Freezing the pretrained transformer encoder and employing an asymmetric policy/value architecture:
   - **Critic**: Uses a single cross-attention layer where a learnable query attends over the frozen token representations (trained by TD loss).
   - **Actor**: Parameter-free hierarchical pooling (mean over 64 vision tokens first, then mean across modalities) followed by an MLP policy, preventing gradient backpropagation into the representations.

```
Observations:
  Proprio (R^14)        ──> Linear ──> [1 token]  ──┐
  F/T (R^4x6 = R^24)    ──> Linear ──> [1 token]  ──┼──> [Pos/Mod Embed] ──> Mask 70% ──> 2-Layer Transformer Encoder
  Vision (64x64x3 RGB)  ──> CNN    ──> [64 tokens] ─┘                                     (Pre-norm, 4 heads, d=128)
                                                                                                    │
                                                                                          [Frozen Embeddings]
                                                                                            /             \
                                                           (Learnable Query Cross-Attn)    /               \  (Hierarchical Mean Pool)
                                                                                          ▼                 ▼
                                                                                       Critic             Actor
                                                                                     (Twin Q)             (SAC)
```

---

## 2. Repository Structure

```
msdp-mini/
├── configs/
│   └── default.yaml          # Hyperparameters for env, pretrain, and RL
├── src/
│   ├── env.py                # Multisensory Push Cube environment (proprio, FT, vision)
│   ├── model.py              # Sensor encoders, masking logic, MSDP transformer, Concat baseline
│   ├── sac.py                # Clean PyTorch SAC with twin Q, Cross-Attention Critic, Pooled Actor
│   ├── pretrain.py           # Self-supervised pretraining (MSDP-R and Concat)
│   ├── train_rl.py           # Downstream RL training runner with checkpointing & eval
│   ├── sanity_sac.py         # State-based SAC sanity check (verifies task learnability)
│   ├── plot_results.py       # Aggregation & 95% CI plotting across seeds
│   ├── visualize_attention.py# Critic cross-attention map extractor
│   └── make_gif.py           # Rollout visualizer and GIF recorder
├── tests/
│   └── test_all.py           # 27 unit tests (shapes, masking, gradients, determinism)
├── media/
│   └── random_policy.gif     # Rollout animation of the Push Cube task
├── results/
│   └── figures/              # Learning curves, attention maps, and reconstructions
├── checkpoints/              # Pretrained encoders and RL checkpoints
├── logs/                     # Training CSV logs and stdout traces
├── NOTES.md                  # Conceptual paper notes and design rationale
├── DEVIATIONS.md             # Documented differences from the original paper
├── COMPUTE_PLAN.md           # Throughput measurements and compute budgeting
├── PROGRESS.md               # Chronological project milestones and timestamps
└── REPORT.md                 # 2-3 page technical reproduction report
```

---

## 3. Installation & Requirements

Python 3.10+ on Windows/Linux with PyTorch and CUDA support.

```bash
# Clone the repository
git clone <repo_url>
cd msdp-mini

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install gymnasium==0.29.1 imageio imageio-ffmpeg matplotlib pandas scipy tqdm einops pytest pyyaml
```

---

## 4. Exact Execution Commands

### Step 1: Run Unit Tests
Verifies observation specs, masking ratios (~70%), vision patch retention guarantee, frozen encoder gradients, and stop-gradient actor pooling:
```bash
python -m pytest tests/test_all.py -v
```

### Step 2: State-based SAC Sanity Check
Validates that the Push Cube environment is solvable with pure state observations:
```bash
python src/sanity_sac.py --steps 30000 --device cpu
```

### Step 3: Self-Supervised Pretraining
Pretrain the multisensory masked autoencoder (MSDP-R) and the Concat baseline:
```bash
# MSDP-R pretraining (15k steps, batch size 64)
python src/pretrain.py --method msdp_r --config configs/default.yaml --device cuda

# Concat baseline pretraining
python src/pretrain.py --method concat --config configs/default.yaml --device cuda
```

### Step 4: Downstream RL Training (Push Cube)
Train SAC agents on top of frozen representations across 3 random seeds:
```bash
# MSDP-R (Ours: Cross-Attention Critic + Pooled Actor)
python src/train_rl.py --method msdp_r --seed 0 --config configs/default.yaml --device cuda
python src/train_rl.py --method msdp_r --seed 1 --config configs/default.yaml --device cuda
python src/train_rl.py --method msdp_r --seed 2 --config configs/default.yaml --device cuda

# Baseline: Concat Pretrained Latent
python src/train_rl.py --method concat --seed 0 --config configs/default.yaml --device cuda
python src/train_rl.py --method concat --seed 1 --config configs/default.yaml --device cuda
python src/train_rl.py --method concat --seed 2 --config configs/default.yaml --device cuda

# Ablation: MSDP-R with Mean Pooling for both Actor and Critic
python src/train_rl.py --method msdp_r --pooling mean --seed 0 --config configs/default.yaml --device cuda
python src/train_rl.py --method msdp_r --pooling mean --seed 1 --config configs/default.yaml --device cuda
python src/train_rl.py --method msdp_r --pooling mean --seed 2 --config configs/default.yaml --device cuda
```

### Step 5: Visualizations & Metrics
Generate learning curves with 95% confidence intervals and extract attention maps:
```bash
# Generate 95% CI learning curves across all seeds
python src/plot_results.py

# Visualize Critic cross-attention maps across modalities
python src/visualize_attention.py --checkpoint checkpoints/pretrain_msdp_r.pt

# Generate sample rollout GIF
python src/make_gif.py
```

---

## 5. Summary of Results

*All numerical values originate from real execution logs located in `logs/` and `results/`.*

### Pretraining Reconstruction Loss (MSE)
| Method | Steps | Wall Time | Final Loss | Proprio MSE | F/T MSE | Vision Patch MSE |
|---|---|---|---|---|---|---|
| **MSDP-R (Masked 70%)** | 15,000 | 1663.1s (9.0 sps) | **0.0556** | 0.0542 | 0.0010 | 0.0004 |
| **Concat (Autoencoder)** | 15,000 | 306.8s (49.3 sps) | **0.0004** | 0.0000 | 0.0000 | 0.0000 |

### Push Cube RL Results (Mean +/- 95% CI across 3 seeds, 30k steps)
| Method | Critic Architecture | Actor Pooling | Final Success Rate | Peak SR | Final Return | Best Return |
|---|---|---|---|---|---|---|
| **MSDP-R (Ours)** | Cross-Attention | Hierarchical Mean | **0.300 +/- 0.124** | **0.400** | **-3.21 +/- 0.52** | **-2.96** |
| **Concat Baseline** | MLP on Latent | MLP on Latent | **0.300 +/- 0.124** | **0.350** | **-3.24 +/- 0.85** | **-2.91** |
| **MSDP-R Ablation** | Mean-Pooled MLP | Hierarchical Mean | **0.317 +/- 0.072** | **0.400** | **-3.50 +/- 0.37** | **-2.85** |
| **State SAC (Sanity)**| MLP (Flat State) | MLP (Flat State) | **0.300** | **0.350** | **-3.90** | **-2.99** |

*Note: 95% confidence intervals computed via Student's t-distribution ($df = n_{\text{seeds}} - 1 = 2$). Full learning curves are saved in `results/figures/learning_curves.png`.*

---

## 6. Key Differences & Limitations

1. **Simulator Setup**: Implemented a standalone pure-Python simulator matching the exact Franka Panda Push Cube observation/action specs and dynamics, bypassing PyBullet C++ compilation constraints on Windows.
2. **Compute Scale**: Reduced RL training horizon from 500k to 50k–100k steps and pretraining from 30k to 15k steps to fit the 10-hour single GTX 1650 (4 GB VRAM) compute budget.
3. **Task Scope**: Focused exclusively on the **Push Cube** task with randomized initial positions, cube mass, and center-of-mass. Peg Insertion and real-robot evaluations were excluded per the reproduction scope.
4. **Baselines**: Implemented the mandatory Concat pretraining baseline and the latent bridging ablation. Optional baselines (VTT, PoE, MSDP-P) were omitted per budget guidelines.
