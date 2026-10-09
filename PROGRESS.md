# PROGRESS LOG: Mini-MSDP Reproduction

**Project:** Independent Reduced-Scope Reproduction of "Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning" (MSDP, arXiv:2511.14427)  
**Author:** MSc AI/ML Student at TU Darmstadt  
**Hardware:** NVIDIA GeForce GTX 1650 (4 GB VRAM), Windows 11, Python 3.10  

---

## Timeline & Milestones

| Timestamp (Local) | Milestone | Status | Notes |
|-------------------|-----------|--------|-------|
| 2026-10-09 10:15 | Paper Reading & Notes | DONE | Read Krohn et al. (arXiv:2511.14427v2); produced `NOTES.md` and initial `DEVIATIONS.md`. |
| 2026-10-09 10:25 | Environment Implementation | DONE | Pure-Python multisensory environment (`PushCubeMultisensoryEnv`) with exact obs shapes (proprio R^14, FT R^(4x6), vision 64x64x3 RGB) & randomizations. Documented mock simulator deviation due to PyBullet MSVC requirements on Windows. |
| 2026-10-09 10:28 | Unit Testing | DONE | 27 pytest unit tests passing (`tests/test_all.py`), validating obs shapes, reset determinism, FT on contact, masking ratio (~70%), vision patch constraint, frozen encoder gradients, and actor stop-gradient. |
| 2026-10-09 10:33 | Benchmark & Compute Plan | DONE | Measured initial throughput (~1800 env steps/s, ~11.6 pretrain iter/s). Reduced pretrain steps from 30k to 15k to fit budget. Wrote initial `COMPUTE_PLAN.md`. |
| 2026-10-09 10:34 | Random Policy Media | DONE | Generated `media/random_policy.gif` (60 frames) showing environment interaction. |
| 2026-10-09 10:40 | Concat Baseline Pretraining | DONE | Completed 15,000 steps in 306.8s (49.3 steps/s) on GPU. Final loss 0.0004. Saved `checkpoints/pretrain_concat.pt` and `results/figures/recon_concat.png`. |
| 2026-10-09 10:50 | State SAC Sanity Check | DONE | Completed 30k steps on CPU. Final SR = 0.300, confirming task learnability. Saved `results/figures/sanity_sac.png` and `logs/sanity_sac_seed0.csv`. |
| 2026-10-09 11:02 | MSDP-R Pretraining | DONE | Completed all 15,000 steps in 1663.1s (9.0 steps/s) on GPU. Final loss 0.0556 (proprio=0.0542, FT=0.0010, vision=0.0004). Saved `checkpoints/pretrain_msdp_r.pt` and `results/figures/recon_msdp_r.png`. |
| 2026-10-09 11:04 | Attention Map Visualization | DONE | Generated `results/figures/attention_maps.png` demonstrating Critic cross-attention distribution across Proprio, F/T, and 8x8 vision patches. |
| 2026-10-09 11:05 | RL Training Batch 1 Launch | IN PROGRESS | Running Seed 0 in parallel on GPU: `msdp_r` (task-375), `concat` (task-377), `ablation_pool` (task-379). |
| 2026-10-09 11:45 | RL Training Batches 2 & 3 | UPCOMING | Execute Seed 1 and Seed 2 for all methods. |
| 2026-10-09 12:30 | Evaluation & Final Plots | UPCOMING | Aggregate learning curves with 95% CI across seeds; compile final results tables in README and REPORT. |

---

## Log of Key Decisions & Adjustments
1. **PyBullet on Windows without MSVC:** Pure-Python environment mimics Panda push dynamics and renders 64x64 visual/tactile observations with identical spec.
2. **Pretraining Steps:** Adjusted to 15k steps based on measured throughput (11.6 it/s = ~22 min) to preserve total compute budget for RL runs.
3. **RL Step Budget:** Empirically measured 17.3 sps during smoke test. 100k steps would require 14.4h (violating compute limit). Budget adjusted to 30k steps per run across 3 seeds, fitting the ~4h RL compute window. Logged in `DEVIATIONS.md` and `COMPUTE_PLAN.md`.
