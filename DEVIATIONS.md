# Deviations from MSDP Paper

All deviations are deliberate design choices due to scope, compute, or simulator limitations.
This is an independent reimplementation; no official code was consulted.

| # | Component | Paper | This Repo | Reason |
|---|-----------|-------|-----------|--------|
| D1 | RL steps | 500k per run | **30k steps per run** (tested in smoke test: 17.3 sps = ~29 min/run) | GTX 1650 4GB VRAM; 9 runs at 100k would take 14.4h, exceeding the hard 4h compute budget |
| D2 | Seeds | 12 | **3** | Time/compute budget |
| D3 | End-effector tool | Round peg held by EE | **Standard panda gripper** (no peg) | Simulator setup complexity; peg model not in panda-gym |
| D4 | FT sensor readings | 4 consecutive sensor readings at different substeps | **4 readings from consecutive PyBullet simulation substeps** stored as ring buffer; exact substep timing not specified in paper — we collect FT at each env step and keep last 4 | Paper silent on exact implementation |
| D5 | MSDP-P | Implemented | **Only MSDP-R implemented** (MSDP-P added if time allows) | Time budget |
| D6 | VTT / PoE baselines | Compared against | **Not implemented** | Time budget; only Concat baseline required |
| D7 | Evaluation frequency | Every epoch (5k steps) | **Every 5k steps** | Same as paper |
| D8 | Eval episodes | Not specified per run | **20 deterministic episodes** | Per task specification |
| D9 | Batch size | 64 | **64** (may reduce to 32 if GPU OOM) | GTX 1650 4GB |
| D10 | Token dim | 128 | **128** | Matches paper |
| D11 | Transformer heads | 4 | **4** | Matches paper |
| D12 | Transformer layers | 2 | **2** | Matches paper |
| D13 | MLP ratio | 2 (hidden=256) | **2** | Matches paper |
| D14 | Masking ratio | 70% | **70%** | Matches paper |
| D15 | Pretraining data | 30k random samples | **30k** | Matches paper |
| D16 | Pretraining steps | 30k | **30k** (reduce to 15k if throughput poor) | Matches paper; fallback stated |
| D17 | Vision tokens | 64 (8×8 from CNN stem) | **64** | Matches paper |
| D18 | Decoder proprio/FT heads | 2-layer MLPs | **Linear heads** (simpler, sufficient for MSE) | Simplification; MLP adds minimal benefit for 1-token sensors |
| D19 | Real-world eval | Yes (Franka robot) | **Simulation only** | No physical robot available |
| D20 | RLPD (real world) | SAC+offline data | **SAC only** | Simulation only |
| D21 | Vision noise (Peg task) | Gaussian noise added | **No noise added** | Push Cube task; paper applies noise to Peg only |
| D22 | Critic cross-attention | Critic trained with cross-attention over frozen embeddings | **Implemented as stated** | Matches paper |
| D23 | Action conditioning (MSDP-P) | Action embedded and fed to decoder | **N/A (MSDP-P skipped)** | D5 |
| D24 | FT ring buffer | Not specified | **Assumption: last 4 env-step FT readings (24 floats) stacked = 4×6** | D4; logged at INFO level |
| D25 | CNN stem channels | Not specified | **[32, 64, 128, 128] with 3×3 convs, stride [2,2,1,1]** → 8×8×128 → linear to 128 | Paper says "4-layer CNN stem similar to [43]"; reference is VideoMAE-v2 |
