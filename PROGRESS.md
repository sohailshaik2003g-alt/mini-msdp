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
| 2026-10-09 10:35 | Pretraining Launched | IN PROGRESS | Concat baseline finished (306.8s, loss 0.0004, `checkpoints/pretrain_concat.pt`, `results/figures/recon_concat.png`). MSDP-R in progress (~55% complete). |
| 2026-10-09 10:50 | State SAC Sanity Check | DONE | Completed 30k steps on CPU. Final SR = 0.300, confirming task learnability. Saved `results/figures/sanity_sac.png` and `logs/sanity_sac_seed0.csv`. |
| 2026-10-09 10:45 | RL Smoke-Testing | DONE | Fixed Windows console cp1252 unicode crash (`\u03b1` -> `alpha`). Measured downstream RL throughput at 17.3 sps. Validated evaluation loop and checkpointing. |
| 2026-10-09 10:47 | Compute Budget Update | DONE | Updated `COMPUTE_PLAN.md` with measured 17.3 sps; adjusted RL horizon to 30,000 steps per run across 3 seeds to fit within 4-hour budget without exceeding time limits. |
| 2026-10-09 10:48 | Documentation & Scripts | DONE | Created `src/visualize_attention.py`, updated `src/plot_results.py`, drafted comprehensive `README.md` and `REPORT.md`. |
| 2026-10-09 11:00 | Downstream RL Training | UPCOMING | Launch 3 methods × 3 seeds (MSDP-R, Concat, Ablation). |
| 2026-10-09 12:30 | Evaluation & Final Plots | UPCOMING | Aggregate learning curves with 95% CI across seeds; generate attention heatmaps. |

---

## Log of Key Decisions & Adjustments
1. **PyBullet on Windows without MSVC:** Pure-Python environment mimics Panda push dynamics and renders 64x64 visual/tactile observations with identical spec.
2. **Pretraining Steps:** Adjusted to 15k steps based on measured throughput (11.6 it/s = ~22 min) to preserve total compute budget for RL runs.
3. **RL Step Budget:** Empirically measured 17.3 sps during smoke test. 100k steps would require 14.4h (violating compute limit). Budget adjusted to 30k steps per run across 3 seeds, fitting the ~4h RL compute window. Logged in `DEVIATIONS.md` and `COMPUTE_PLAN.md`.
