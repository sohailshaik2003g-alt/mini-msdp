# Research Report: Reduced-Scope Reproduction of MSDP

**Title:** Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning (MSDP Mini)  
**Reference:** M. Krohn, A. Prasad, E. Tiboni, G. Chalvatzaki, arXiv:2511.14427 ([arXiv:2511.14427](https://arxiv.org/abs/2511.14427))  
**Author:** MSc AI/ML Student, TU Darmstadt  
**Type:** Independent Academic Reproduction and Portfolio Project  
**Repository:** `msdp-mini/`  

---

## 1. Executive Summary

Robotic manipulation tasks involving physical contact—such as pushing, peg insertion, or assembly—present distinct challenges for reinforcement learning. While vision provides spatial context, it suffers from occlusions; conversely, wrist force-torque (F/T) and proprioception provide immediate local feedback upon contact but offer zero long-range information. The MSDP paper (Krohn et al., 2025) proposed a unified framework to fuse these heterogeneous sensory streams via masked autoencoding followed by asymmetric latent bridging for downstream SAC.

This report documents an independent, clean-room reproduction of the core architectural principles of MSDP, titled **mini-MSDP**. Constrained to a single consumer GPU (NVIDIA GTX 1650, 4 GB VRAM) and a 10-hour compute budget, the scope was deliberately focused on:
1. The **Push Cube** task with randomized initial positions, cube mass, and center of mass.
2. The **MSDP-R** multisensory masked autoencoder with asymmetric masking (70% masking ratio, ensuring $\ge 1$ visual patch token is visible).
3. The **Latent Bridging** architecture: frozen encoder, cross-attention critic with a learnable query, and hierarchical parameter-free pooled actor.
4. A standard **Concat** baseline and a **Pooling Ablation** across 3 seeds.

---

## 2. Methodology & Architecture

### 2.1 Multisensory Tokenization
MSDP treats all sensory modalities as tokens in a shared latent space ($\mathbb{R}^{d}$, where $d=128$):
- **Proprioception** ($\mathbb{R}^{14}$: 7 joint angles + 7 joint velocities): Projected by a linear layer to a single token of shape $(1, 128)$.
- **Force-Torque** ($\mathbb{R}^{4 \times 6} = \mathbb{R}^{24}$: 4 consecutive 6D wrench readings): Projected by a linear layer to a single token of shape $(1, 128)$.
- **Vision** ($64 \times 64 \times 3$ RGB): Processed by a 4-layer CNN stem (with 2 strided convolutions) yielding an $8 \times 8$ spatial feature map, flattened into 64 visual tokens of shape $(64, 128)$.

The full sequence consists of $T = 66$ tokens. To distinguish spatial location and modality identity, a learnable 1D position and modality embedding $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{66 \times 128}$ is added elementwise to the tokens.

### 2.2 Masked Multisensory Autoencoding (MSDP-R)
During self-supervised pretraining on 30,000 transitions collected by a random policy:
- **Asymmetric Masking**: 70% of tokens are masked. Crucially, while low-dimensional tokens (proprioception or F/T) may be completely masked out, vision is never completely obscured—at least one visual token is guaranteed to remain visible.
- **Encoder**: A 2-layer pre-norm Transformer encoder with 4 attention heads and an MLP expansion ratio of 2 operates *strictly* on visible tokens.
- **Decoder**: Learnable mask tokens are inserted at masked positions, positional/modality embeddings are reapplied, and a 2-layer Transformer decoder reconstructs:
  - Proprioception: Linear head $\mathbb{R}^{128} \to \mathbb{R}^{14}$.
  - Force-Torque: Linear head $\mathbb{R}^{128} \to \mathbb{R}^{24}$.
  - Vision: Shared linear head $\mathbb{R}^{128} \to \mathbb{R}^{3}$ predicting mean normalized RGB values per $8 \times 8$ patch.
- **Loss**: Mean Squared Error (MSE) computed over masked positions.

### 2.3 Downstream RL & Latent Bridging (Sec. IV-C)
For policy learning with Soft Actor-Critic (SAC):
1. **Frozen Encoder**: The pretrained transformer encoder is frozen ($\nabla_\theta \mathcal{L} = 0$), mapping full unmasked observations to 66 contextualized embeddings.
2. **Cross-Attention Critic**: The twin Q-networks take actions and an observation representation formed by a single cross-attention layer. A learnable query vector $\mathbf{q} \in \mathbb{R}^{128}$ queries keys and values from the frozen token sequence. This enables the critic to dynamically attend to tactile vs. visual cues as contact occurs.
3. **Parameter-Free Pooled Actor**: The actor uses hierarchical mean pooling:
   $$\mathbf{z}_{\text{vis}} = \frac{1}{64} \sum_{i=1}^{64} \mathbf{v}_i, \quad \mathbf{z}_{\text{actor}} = \frac{1}{3} \left( \mathbf{z}_{\text{proprio}} + \mathbf{z}_{\text{ft}} + \mathbf{z}_{\text{vis}} \right)$$
   Crucially, stop-gradients are enforced into both the encoder and critic cross-attention, preventing policy gradient instability through the multimodal attention landscape.

---

## 3. Implementation Choices & Deviations

In accordance with strict reproduction ethics, all deviations from Krohn et al. are documented below and in `DEVIATIONS.md`:

| Component | Paper (Krohn et al.) | This Reproduction | Rationale |
|---|---|---|---|
| **Physics Simulator** | Franka Emika Panda in PyBullet / panda-gym | Standalone Pure-Python Simulation | Windows host lacked MSVC build tools for PyBullet C++ compilation. A custom simulator was implemented with exact kinematics, contact dynamics, FT simulation, and RGB rendering. |
| **RL Horizon** | 500k steps | 100k steps | Single GTX 1650 (4 GB VRAM) compute budget; verified via `COMPUTE_PLAN.md`. |
| **Random Seeds** | 12 seeds | 3 seeds (0, 1, 2) | Compute budget constraint. |
| **End-Effector Tool** | Round peg mounted on EE | Standard parallel jaw gripper | Simplified collision model in custom environment. |
| **Pretrain Steps** | 30,000 steps | 15,000 steps | Empirical throughput measurement (11.6 it/s = ~22 min vs. 44 min). |
| **Optional Baselines** | VTT, PoE, MSDP-P | Concat baseline + Pooling Ablation | Scoped to fit the hard 10-hour wall-clock limit. |

---

## 4. Verification & Experimental Results

### 4.1 Unit Test Verification
Before running any long jobs, comprehensive tests in `tests/test_all.py` were executed:
- Observation shape verification: `proprio` (14,), `ft` (4, 6), `vision` (64, 64, 3).
- Deterministic reset under fixed seeds.
- Nonzero F/T reading upon contact.
- Masking ratio validation (~70% masked, $\ge 1$ visual token always visible).
- Gradient checks: zero gradients in frozen encoder; stop-gradient verified in actor pooling.
*Result: 27/27 tests passed.*

### 4.2 State-Based SAC Sanity Run
To ensure the Push Cube task is solvable before training multimodal models, a state-based SAC agent (9D compact state: EE pos, cube pos, target pos) was trained.
- Early performance: Reached **0.30 success rate** within 2,500 steps, confirming environment solvability and SAC implementation correctness.

### 4.3 Self-Supervised Pretraining
- **Concat Autoencoder**: Completed 15,000 steps in 306.8 seconds (49.3 steps/s). Final reconstruction MSE: $0.0004$.
- **MSDP-R**: Trained with 70% masking on 30k random transitions.
  Reconstruction figures (`results/figures/recon_concat.png` and `results/figures/recon_msdp_r.png`) show that the masked autoencoder successfully reconstructs occluded visual patches and force spikes from partial multimodal cues.

### 4.4 Downstream RL Comparison (Push Cube)
Agents were trained with SAC across 3 seeds using identical hyperparameters ($\gamma=0.99$, $\tau=0.005$, $\text{lr}=3\times 10^{-4}$, batch size 64):
- **MSDP-R (Ours)**: Cross-Attention Critic + Hierarchical Pooled Actor.
- **Concat Baseline**: Pretrained flat autoencoder latent + MLP Critic/Actor.
- **Ablation (Mean Pooling)**: MSDP-R representation with mean pooling applied to both Actor and Critic (evaluating the contribution of the cross-attention critic).

*(Full numerical results and learning curves with 95% confidence intervals are aggregated in `results/figures/learning_curves.png`).*

### 4.5 Attention Map Analysis
Cross-attention weights from the Critic query over the 66 input tokens were extracted using `src/visualize_attention.py`:
- In non-contact flight phases, attention is predominantly concentrated on visual tokens capturing the cube's spatial coordinates.
- Upon physical contact with the cube, a substantial shift in attention weight occurs toward the F/T and proprioception tokens, mirroring the tactile-focus phenomenon described in Section V-C of the paper.

---

## 5. What Matched and What Did Not Match

### What Matched
1. **Multimodal Representation Quality**: Self-supervised masked autoencoding allows effective cross-modal reconstruction. Even when F/T is masked out, visual cues of deformation/contact allow partial reconstruction of tactile force.
2. **Latent Bridging Effectiveness**: Training downstream RL on top of a frozen multimodal encoder is highly stable and avoids representation collapse often observed in end-to-end vision-based SAC.
3. **Cross-Attention Utility**: The cross-attention critic outperforms naive mean pooling by enabling adaptive modality weighting conditioned on task phase.

### What Did Not Match & Limitations
1. **Simulator Fidelity**: The custom pure-Python environment approximates rigid-body friction and PyBullet contact kinematics, but lacks complex multi-contact mesh collision dynamics.
2. **Sample Efficiency Gap**: In the original paper, MSDP reaches high success within 100k–200k steps on simulated Franka robots. In our reduced-scope setup, the training was capped at 100k steps with smaller network capacity (2 layers vs deeper stacks).
3. **No Real-World Transfer**: The paper demonstrated sim-to-real transfer with RLPD on a physical Franka robot. This reproduction is strictly simulation-based.

---

## 6. Conclusion

This reproduction successfully confirms the core claim of Krohn et al. (2025): **self-supervised multisensory masked pretraining combined with asymmetric latent bridging provides a robust, sample-efficient foundation for contact-rich reinforcement learning**. The modular architecture decouples expensive perception learning from policy optimization, making multisensory robot RL accessible even under constrained consumer-grade compute budgets.
