# MSDP Paper Notes (own words, ~15 min read)

**Paper**: "Self-Supervised Multisensory Pretraining for Contact-Rich Robot RL" (Krohn et al., arXiv:2511.14427)

## Core Idea
Two-phase framework:
1. **Pretraining** (offline, random-policy data): Masked Autoencoder on multisensory tokens. Mask 70% of tokens, reconstruct from remaining. Cross-modal prediction forces sensor fusion.
2. **Downstream RL** (online SAC): Frozen encoder. Asymmetric critic/actor bridging.

## Architecture (Sec. IV)

### Sensor Encoders
- **Vision** (64×64×3 → 8×8 grid = 64 tokens, each 128-dim): 4-layer CNN stem. CNN gives overlapping receptive fields → stabilizes masking (masked positions aren't recoverable pixel-by-pixel anyway).
- **Proprio** (14-dim) → linear projection → 1 token, 128-dim
- **FT** (4×6=24-dim) → linear projection → 1 token, 128-dim  
- **Total tokens**: 66 (64 vision + 1 proprio + 1 FT)

### Position/Modality Embeddings
- Learnable (not fixed sinusoidal). Each token gets one embedding encoding both its position AND which sensor it came from.

### Encoder (Transformer)
- 2 layers, 4 attention heads, pre-norm LayerNorm, MLP ratio 2 (hidden=256)
- Only visible tokens (30% of 66 = ~20 tokens) enter encoder
- Vision is never fully masked (keeps at least 1 vision token visible)

### Decoder
- Learnable mask token added back at masked positions
- Re-apply position/modality embeddings
- 2-layer transformer (same dims)
- Heads: separate linear for proprio/FT; shared linear for all vision patches (8×8×3 = 192 output dims per vision token)

### Objectives
- **MSDP-R**: reconstruct current obs O_t (MSE). No action conditioning.
- **MSDP-P**: reconstruct next obs O_{t+1} conditioned on action A_t (action embedded and fed to decoder).

## Latent Bridging (Sec. IV-C) — KEY DESIGN CHOICE
After pretraining, encoder frozen. How do you extract a compact state for RL?

- **Critic**: Single cross-attention layer. Query = learnable (trained by critic loss). Keys+Values = frozen encoder embeddings (66 tokens). Output = task-specific compact rep. Dynamic throughout training!
- **Actor**: Parameter-free mean pooling. First average all 64 vision tokens → 1, then mean over 3 tokens (vision-pooled + proprio + FT) → 128-dim. Stable, no extra params.
- **Why asymmetric?** Actor stability > actor expressiveness. Critic expressiveness > critic stability. Matches TD3 paper observation that critic can use extra info actor can't.

## Masking Strategy (important!)
- 70% of tokens masked overall
- Vision tokens: never ALL masked (always ≥1 visible)
- Low-dim tokens (proprio, FT): may be fully masked (they're more information-dense; vision can still predict them)
- Only visible tokens go into encoder (not zero-masked like BERT, truly removed like MAE)

## Baselines
- **Concat**: CNN+MLP per sensor → concatenate → 2-layer MLP → latent. Pretrained with full reconstruction (no masking), frozen for RL.
- **VTT**: Vision Transformer Transformer — transformer-based fusion.
- **PoE**: Product of Experts — separate Gaussian per sensor, multiply → shared latent.

## Experimental Numbers (sanity reference only)
- 30k random policy samples for pretraining
- 30k update steps for pretraining
- 500k RL steps (we use 100k — DEVIATION)
- 12 seeds reported (we use 3 — DEVIATION)
- Push Cube: MSDP-R/P learn fast, baselines need longer (steeper curve for MSDP)

## Key Implementation Choices to Log
- FT sensor: 4 consecutive readings simulated via PyBullet substeps (assumption, see DEVIATIONS.md)
- Round peg: Not implemented, use standard gripper (DEVIATION)
- MSDP-P: Implement if time allows after MSDP-R
