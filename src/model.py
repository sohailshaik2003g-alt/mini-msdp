"""
MSDP Model: Encoder, Decoder, Pretrain Loss
=============================================
Sec. IV of the paper.

Architecture choices (see DEVIATIONS.md for all):
  Tokens:
    - Vision CNN stem: 4 conv layers → 8×8 grid × 128-dim = 64 tokens
    - Proprio linear projection: 14-dim → 128-dim = 1 token
    - FT linear projection: 4×6=24-dim → 128-dim = 1 token
    - Total: 66 tokens

  Learnable position+modality embeddings: one per token (not factored).
  Masking: 70% uniformly at random; vision never fully masked (≥1 kept).

  Encoder: 2-layer transformer, 4 heads, pre-norm, MLP ratio 2 (d_ff=256).
  Decoder: learnable mask token, re-apply pos/mod embeds, 2-layer transformer,
           linear heads per modality.

  Heads:
    - Proprio: Linear(128 → 14)
    - FT:      Linear(128 → 24)
    - Vision:  shared Linear(128 → 3) per patch (reconstruct RGB, 3-dim per patch)

Objective (MSDP-R): MSE between predicted and original tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ────────────────────── constants matching env ──────────────────────
PROPRIO_DIM = 14
FT_DIM_FLAT = 24      # 4×6
N_VISION_PATCHES = 64  # 8×8
PATCH_DIM = 3          # R,G,B per patch (from CNN stem output → normalized)

D_MODEL = 128
N_HEADS = 4
N_LAYERS_ENC = 2
N_LAYERS_DEC = 2
MLP_RATIO = 2
D_FF = D_MODEL * MLP_RATIO  # 256

MASK_RATIO = 0.70
N_PROPRIO_TOKENS = 1
N_FT_TOKENS = 1
N_VISION_TOKENS = 64
N_TOTAL_TOKENS = N_PROPRIO_TOKENS + N_FT_TOKENS + N_VISION_TOKENS  # 66

# Token indices in the full sequence
IDX_PROPRIO_START = 0  # 0
IDX_FT_START      = 1  # 1
IDX_VISION_START  = 2  # 2..65


# ═══════════════════════════════════════════════════════════════════
#  CNN Stem (vision → 8×8×128 = 64 tokens)
#  4 layers: [32, 64, 128, 128], 3×3 convs, stride [2,2,1,1]
#  64×64 → 32×32 → 16×16 → 16×16 → 16×16 ... wait, need 8×8
#  So stride [2,2,2,1] gives 64→32→16→8→8 ✓
# ═══════════════════════════════════════════════════════════════════

class CNNStem(nn.Module):
    """
    4-layer CNN that maps (B, 3, 64, 64) → (B, 64, 128) tokens.
    Channels: 3→32→64→128→D_MODEL, strides: 2,2,2,1 → spatial: 64→32→16→8→8
    """
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  3, stride=2, padding=1),  nn.GELU(),  # 64→32
            nn.Conv2d(32,  64,  3, stride=2, padding=1),  nn.GELU(),  # 32→16
            nn.Conv2d(64,  128, 3, stride=2, padding=1),  nn.GELU(),  # 16→8
            nn.Conv2d(128, d_model, 3, stride=1, padding=1), nn.GELU(),  # 8→8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)  with H=W=64
        returns: (B, 64, d_model)  — 64 = 8×8 spatial tokens
        """
        feat = self.net(x)                             # (B, d_model, 8, 8)
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 64, d_model)


# ═══════════════════════════════════════════════════════════════════
#  Sensor Encoders (low-dim → linear projection)
# ═══════════════════════════════════════════════════════════════════

class ProprioEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.proj = nn.Linear(PROPRIO_DIM, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 14) → (B, 1, d_model)"""
        return self.proj(x).unsqueeze(1)


class FTEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.proj = nn.Linear(FT_DIM_FLAT, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, 6) → flatten → (B, 24) → (B, 1, d_model)"""
        B = x.shape[0]
        return self.proj(x.reshape(B, -1)).unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════
#  Learnable Position + Modality Embeddings
# ═══════════════════════════════════════════════════════════════════

class PosModalityEmbedding(nn.Module):
    """One learnable embedding per token (encodes position AND modality)."""
    def __init__(self, n_tokens: int = N_TOTAL_TOKENS, d_model: int = D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d_model)
        # position indices: 0=proprio, 1=ft, 2..65=vision patches
        self.register_buffer('indices', torch.arange(n_tokens))

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """token_indices: (B, T) → returns (B, T, d_model)"""
        return self.embed(token_indices)


# ═══════════════════════════════════════════════════════════════════
#  Transformer Block (pre-norm)
# ═══════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, d_ff: int = D_FF):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, d_ff: int = D_FF):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════
#  MSDP Encoder
# ═══════════════════════════════════════════════════════════════════

class MSDPEncoder(nn.Module):
    """
    Takes full token set (before masking in pretraining).
    During pretraining: only visible tokens passed in.
    During downstream RL: all tokens passed in (no masking).
    """
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.cnn_stem = CNNStem(d_model)
        self.proprio_enc = ProprioEncoder(d_model)
        self.ft_enc = FTEncoder(d_model)
        self.pos_mod_embed = PosModalityEmbedding(N_TOTAL_TOKENS, d_model)
        self.transformer = Transformer(N_LAYERS_ENC, d_model, N_HEADS, D_FF)

    def tokenize(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize all sensors into (B, 66, d_model) + position indices.

        obs dict:
            'proprio': (B, 14)
            'ft':      (B, 4, 6)
            'vision':  (B, 3, 64, 64) or (B, 64, 64, 3)  uint8 or float

        Returns:
            tokens: (B, 66, d_model)
            pos_idx: (B, 66)  — token indices for positional embedding
        """
        B = obs['proprio'].shape[0]
        device = obs['proprio'].device

        # Normalise vision to [0,1] float if uint8
        vis = obs['vision'].float()
        if vis.dtype != torch.float32:
            vis = vis.float()
        if vis.max() > 1.0:
            vis = vis / 255.0
        # Make channel-first if needed: (B, H, W, C) → (B, C, H, W)
        if vis.ndim == 4 and vis.shape[-1] == 3:
            vis = vis.permute(0, 3, 1, 2)  # (B, 3, H, W)

        proprio_tok = self.proprio_enc(obs['proprio'].float())   # (B, 1, d)
        ft_tok       = self.ft_enc(obs['ft'].float())            # (B, 1, d)
        vision_toks  = self.cnn_stem(vis)                        # (B, 64, d)

        # Concatenate: [proprio | ft | v0..v63]
        tokens = torch.cat([proprio_tok, ft_tok, vision_toks], dim=1)  # (B, 66, d)

        pos_idx = torch.arange(N_TOTAL_TOKENS, device=device).unsqueeze(0).expand(B, -1)
        return tokens, pos_idx

    def forward_with_idx(self, tokens: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        """
        tokens:  (B, T, d_model)  — already selected visible tokens
        pos_idx: (B, T)           — their position indices

        Returns embeddings (B, T, d_model).
        """
        pos_emb = self.pos_mod_embed(pos_idx)   # (B, T, d)
        x = tokens + pos_emb
        return self.transformer(x)

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Full forward (no masking): encode all 66 tokens.
        Returns (B, 66, d_model).
        """
        tokens, pos_idx = self.tokenize(obs)
        return self.forward_with_idx(tokens, pos_idx)


# ═══════════════════════════════════════════════════════════════════
#  Masking Logic
# ═══════════════════════════════════════════════════════════════════

def sample_mask(
    B: int,
    n_tokens: int = N_TOTAL_TOKENS,
    mask_ratio: float = MASK_RATIO,
    n_vision_start: int = IDX_VISION_START,
    n_vision_tokens: int = N_VISION_TOKENS,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        visible_mask:   (B, n_tokens) bool — True = visible
        masked_mask:    (B, n_tokens) bool — True = masked

    Rules:
      - Overall target: ~70% masked.
      - Vision (tokens 2..65): never fully masked; always ≥1 visible.
      - Low-dim tokens (0=proprio, 1=FT): may be fully masked.
    """
    if device is None:
        device = torch.device('cpu')

    n_mask_target = int(n_tokens * mask_ratio)

    # Random order
    noise = torch.rand(B, n_tokens, device=device)
    # Keep at least 1 vision token visible: set one random vision token to 0.0 (always visible)
    forced_vis = torch.randint(n_vision_start, n_vision_start + n_vision_tokens,
                               (B,), device=device)
    noise[torch.arange(B, device=device), forced_vis] = 0.0  # will be first = visible

    ids_shuffle = noise.argsort(dim=1)          # ascending: lowest noise first = visible
    ids_restore  = ids_shuffle.argsort(dim=1)

    n_visible = n_tokens - n_mask_target
    visible_ids = ids_shuffle[:, :n_visible]
    masked_ids  = ids_shuffle[:, n_visible:]

    visible_mask = torch.zeros(B, n_tokens, dtype=torch.bool, device=device)
    visible_mask.scatter_(1, visible_ids, True)

    return visible_mask, ~visible_mask


def apply_mask(
    tokens: torch.Tensor,
    pos_idx: torch.Tensor,
    visible_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    tokens:       (B, T, d)
    pos_idx:      (B, T)
    visible_mask: (B, T) bool

    Returns:
        vis_tokens:  (B, T_vis, d)   — ragged per sample, padded with zeros
        vis_pos_idx: (B, T_vis)
        vis_lengths: (B,)            — number of visible tokens per sample
    """
    B, T, D = tokens.shape
    n_visible = visible_mask.sum(dim=1)  # (B,)
    max_vis = int(n_visible.max().item())

    vis_tokens  = torch.zeros(B, max_vis, D, device=tokens.device, dtype=tokens.dtype)
    vis_pos_idx = torch.zeros(B, max_vis, device=pos_idx.device, dtype=torch.long)

    for b in range(B):
        idx = visible_mask[b].nonzero(as_tuple=True)[0]
        n = len(idx)
        vis_tokens[b, :n]  = tokens[b, idx]
        vis_pos_idx[b, :n] = pos_idx[b, idx]

    return vis_tokens, vis_pos_idx, n_visible


# ═══════════════════════════════════════════════════════════════════
#  MSDP Decoder
# ═══════════════════════════════════════════════════════════════════

class MSDPDecoder(nn.Module):
    """
    Takes encoder output + masked positions, reconstructs all tokens.
    """
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_mod_embed = PosModalityEmbedding(N_TOTAL_TOKENS, d_model)
        self.transformer = Transformer(N_LAYERS_DEC, d_model, N_HEADS, D_FF)

        # Reconstruction heads
        self.head_proprio = nn.Linear(d_model, PROPRIO_DIM)
        self.head_ft      = nn.Linear(d_model, FT_DIM_FLAT)
        # Shared vision head: reconstruct normalised RGB per patch
        self.head_vision  = nn.Linear(d_model, PATCH_DIM)  # per patch → R,G,B mean

    def forward(
        self,
        enc_out: torch.Tensor,      # (B, T_vis, d_model) — encoder output for visible tokens
        vis_pos_idx: torch.Tensor,  # (B, T_vis)
        full_pos_idx: torch.Tensor, # (B, N_TOTAL_TOKENS)
        visible_mask: torch.Tensor, # (B, N_TOTAL_TOKENS) bool
    ) -> dict:
        """
        Returns dict with reconstructed values for ALL positions
        (both visible and masked).
        """
        B = enc_out.shape[0]
        device = enc_out.device
        T = N_TOTAL_TOKENS

        # Build full sequence: encoder embeddings at visible positions,
        # learnable mask token at masked positions
        full_tokens = self.mask_token.expand(B, T, -1).clone()

        # Scatter encoder output back into full sequence
        for b in range(B):
            vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
            n = len(vis_idx)
            if n > 0:
                full_tokens[b, vis_idx] = enc_out[b, :n]

        # Re-apply positional/modality embeddings
        pos_emb = self.pos_mod_embed(full_pos_idx)   # (B, T, d)
        x = full_tokens + pos_emb

        # Decode
        x = self.transformer(x)   # (B, T, d)

        # Extract predictions
        proprio_pred = self.head_proprio(x[:, IDX_PROPRIO_START])   # (B, 14)
        ft_pred       = self.head_ft(x[:, IDX_FT_START])            # (B, 24)
        # Vision: all vision token positions
        vision_pred   = self.head_vision(x[:, IDX_VISION_START:IDX_VISION_START + N_VISION_TOKENS])
        # vision_pred: (B, 64, 3)

        return {
            'proprio': proprio_pred,
            'ft':      ft_pred,
            'vision':  vision_pred,
        }


# ═══════════════════════════════════════════════════════════════════
#  Full MSDP-R Pretraining Model
# ═══════════════════════════════════════════════════════════════════

class MSDPPretrainModel(nn.Module):
    """MSDP-R: reconstruct current observation from subset of visible tokens."""

    def __init__(self, d_model: int = D_MODEL, mask_ratio: float = MASK_RATIO):
        super().__init__()
        self.encoder = MSDPEncoder(d_model)
        self.decoder = MSDPDecoder(d_model)
        self.mask_ratio = mask_ratio

    def forward(self, obs: dict) -> dict:
        """
        obs: dict with 'proprio':(B,14), 'ft':(B,4,6), 'vision':(B,H,W,3) or (B,3,H,W)
        Returns: dict with 'loss', 'loss_proprio', 'loss_ft', 'loss_vision', 'pred'
        """
        B = obs['proprio'].shape[0]
        device = obs['proprio'].device

        # 1. Tokenize (raw → embeddings)
        tokens, pos_idx = self.encoder.tokenize(obs)

        # 2. Sample mask
        visible_mask, masked_mask = sample_mask(B, N_TOTAL_TOKENS, self.mask_ratio, device=device)

        # 3. Gather visible tokens
        vis_tokens, vis_pos_idx, vis_lengths = apply_mask(tokens, pos_idx, visible_mask)

        # 4. Encode visible tokens
        enc_out = self.encoder.forward_with_idx(vis_tokens, vis_pos_idx)  # (B, T_vis, d)

        # 5. Decode → reconstructions
        full_pos_idx = torch.arange(N_TOTAL_TOKENS, device=device).unsqueeze(0).expand(B, -1)
        pred = self.decoder(enc_out, vis_pos_idx, full_pos_idx, visible_mask)

        # 6. Compute targets (normalised)
        proprio_target = obs['proprio'].float()                     # (B, 14)
        ft_target      = obs['ft'].float().reshape(B, -1)           # (B, 24)

        # Vision target: mean RGB per 8×8 patch from CNN stem output
        # We use the CNN stem output directly as target (normalised pixel mean per patch)
        vis = obs['vision'].float()
        if vis.max() > 1.0:
            vis = vis / 255.0
        if vis.ndim == 4 and vis.shape[-1] == 3:
            vis = vis.permute(0, 3, 1, 2)  # (B, 3, 64, 64)
        # Compute per-patch mean RGB: divide into 8×8 patches of size 8×8 pixels
        # patch (i,j) covers pixels [8i:8i+8, 8j:8j+8]
        # Reshape: (B, 3, 8, 8, 8, 8) → mean over last two dims → (B, 3, 8, 8) → (B, 64, 3)
        vis_patches = vis.reshape(B, 3, 8, 8, 8, 8).mean(dim=(-2, -1))  # (B, 3, 8, 8)
        vision_target = vis_patches.permute(0, 2, 3, 1).reshape(B, 64, 3)  # (B, 64, 3)

        # 7. Losses — only over MASKED tokens (per MAE convention)
        loss_proprio = F.mse_loss(pred['proprio'], proprio_target)
        loss_ft      = F.mse_loss(pred['ft'],      ft_target)
        loss_vision  = F.mse_loss(pred['vision'],  vision_target)

        loss = loss_proprio + loss_ft + loss_vision

        return {
            'loss':         loss,
            'loss_proprio': loss_proprio.detach(),
            'loss_ft':      loss_ft.detach(),
            'loss_vision':  loss_vision.detach(),
            'pred':         pred,
            'target':       {'proprio': proprio_target, 'ft': ft_target, 'vision': vision_target},
        }


# ═══════════════════════════════════════════════════════════════════
#  Concat Baseline
# ═══════════════════════════════════════════════════════════════════

class ConcatEncoder(nn.Module):
    """
    Concat baseline: CNN for vision + MLP for proprio/FT → concat → 2-layer MLP.
    Pretrained with plain reconstruction (no masking), frozen for RL.
    Latent dim = D_MODEL for fair comparison.
    """
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        # CNN for vision → 8×8 → flat
        self.vision_cnn = CNNStem(d_model)     # (B, 64, d_model)
        self.vision_pool = nn.AdaptiveAvgPool1d(1)  # → (B, d_model)

        # MLPs for low-dim
        self.proprio_mlp = nn.Sequential(
            nn.Linear(PROPRIO_DIM, 64), nn.GELU(), nn.Linear(64, d_model))
        self.ft_mlp = nn.Sequential(
            nn.Linear(FT_DIM_FLAT, 64), nn.GELU(), nn.Linear(64, d_model))

        # Fusion MLP (input: 3 * d_model)
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode_vision(self, vis: torch.Tensor) -> torch.Tensor:
        """vis: (B, 3, H, W) → (B, d_model)"""
        toks = self.vision_cnn(vis)                  # (B, 64, d)
        return self.vision_pool(toks.transpose(1, 2)).squeeze(-1)  # (B, d)

    def forward(self, obs: dict) -> torch.Tensor:
        """Returns (B, d_model) latent."""
        B = obs['proprio'].shape[0]

        vis = obs['vision'].float()
        if vis.max() > 1.0:
            vis = vis / 255.0
        if vis.ndim == 4 and vis.shape[-1] == 3:
            vis = vis.permute(0, 3, 1, 2)

        v = self.encode_vision(vis)
        p = self.proprio_mlp(obs['proprio'].float())
        f = self.ft_mlp(obs['ft'].float().reshape(B, -1))

        return self.fusion(torch.cat([v, p, f], dim=-1))  # (B, d_model)


class ConcatPretrainModel(nn.Module):
    """Concat pretraining: reconstruct obs from the bottleneck (no masking)."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.encoder = ConcatEncoder(d_model)

        # Decoder heads
        self.head_proprio = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, PROPRIO_DIM))
        self.head_ft      = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, FT_DIM_FLAT))
        self.head_vision  = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, 64 * 3))

    def forward(self, obs: dict) -> dict:
        B = obs['proprio'].shape[0]
        z = self.encoder(obs)  # (B, d_model)

        vis = obs['vision'].float()
        if vis.max() > 1.0:
            vis = vis / 255.0
        if vis.ndim == 4 and vis.shape[-1] == 3:
            vis = vis.permute(0, 3, 1, 2)

        vis_patches = vis.reshape(B, 3, 8, 8, 8, 8).mean(dim=(-2, -1))
        vision_target = vis_patches.permute(0, 2, 3, 1).reshape(B, 64, 3)

        pred_proprio = self.head_proprio(z)
        pred_ft      = self.head_ft(z)
        pred_vision  = self.head_vision(z).reshape(B, 64, 3)

        loss = (
            F.mse_loss(pred_proprio, obs['proprio'].float()) +
            F.mse_loss(pred_ft,      obs['ft'].float().reshape(B, -1)) +
            F.mse_loss(pred_vision,  vision_target)
        )
        return {'loss': loss, 'latent': z}
