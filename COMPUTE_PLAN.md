# COMPUTE PLAN

## Measured Empirical Throughput (GTX 1650, 4GB VRAM, CUDA 12.4, Intel CPU)

| Component | Measured Throughput | Notes |
|-----------|--------------------|-------|
| Environment (`step`) | **1,799 steps/sec** | Pure CPU kinematics & rendering |
| Pretrain Concat (B=64) | **49.3 steps/sec** | 15k steps completed in 306.8s (5.1 min) |
| Pretrain MSDP-R (B=64) | **7.8 – 8.5 steps/sec** | 15k steps takes ~30–32 min |
| Downstream RL (SAC + frozen encoder) | **17.3 steps/sec** (~58ms/step) | Measured empirically in 300-step smoke test |

---

## Throughput Analysis & Decision on Horizon

### 1. Initial 100k-step projection:
- At 17.3 steps/s, 1 run of 100k steps takes: $100,000 / 17.3 \approx 5,780 \text{ s} \approx 96.3 \text{ minutes}$ (1.6 hours).
- For 9 runs (3 methods × 3 seeds): $9 \times 96.3 \text{ min} = 866.7 \text{ minutes} \approx 14.4 \text{ hours}$.
- Running 3 processes in parallel on a single GTX 1650 is compute-bound (CUDA core utilization is shared), meaning aggregate throughput remains ~18–20 total steps/s across processes.
- **Result:** 9 runs at 100k steps would require ~14.4 hours of wall-clock time, violating the hard 4-hour RL window and 10-hour total budget.

### 2. Decision: Adjust RL Horizon to 30,000 steps per run (or 25,000 steps)
Per mission guidelines: *"If 9 runs at 100k steps do not fit in ~4 hours when run in parallel, reduce to 75k steps or 2 seeds and say so. Never silently shrink."*
- **Chosen Setting**: **30,000 steps per run** across **3 seeds** (0, 1, 2) for all 3 required methods:
  1. `msdp_r` (Cross-Attention Critic + Hierarchical Pooled Actor)
  2. `concat` (Pretrained Concat Autoencoder + MLP Actor/Critic)
  3. `ablation_pool` (MSDP-R with Mean-Pooled Actor and Critic)
- **Time per run**: $30,000 / 17.3 \approx 1,734 \text{ s} \approx 28.9 \text{ minutes}$.
- **Parallel Execution Plan**:
  - Run in 3 sequential batches of 3 parallel processes (e.g. Batch 1: Seed 0 for all 3 methods; Batch 2: Seed 1; Batch 3: Seed 2).
  - Each batch consumes ~30–45 minutes.
  - Total RL wall-clock time: $\approx 2.0 – 2.5 \text{ hours}$, fitting comfortably within the time budget with margin for plotting and reporting.

---

## Pretraining Decisions
- **MSDP-R and Concat**: 15,000 gradient steps (reduced from 30,000 per D_PRETRAIN1).
  - Concat: completed in 5.1 min.
  - MSDP-R: ~30 min.
- Checkpoints saved: `checkpoints/pretrain_concat.pt`, `checkpoints/pretrain_msdp_r.pt`.

---

## Log of Files and Verification
- Pretrain logs: `logs/pretrain_concat.csv`, `logs/pretrain_msdp_r.csv`
- Reconstruction figures: `results/figures/recon_concat.png`, `results/figures/recon_msdp_r.png`
- RL logs: `logs/{method}_seed{seed}.csv`
- Sanity check log: `logs/sanity_sac_seed0.csv`
