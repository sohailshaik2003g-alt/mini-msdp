"""
SAC (Soft Actor-Critic) with twin Q-critics.
CleanRL-style compact implementation.

Reference: Haarnoja et al., SAC (2018).

Two usage modes:
  1. State-based SAC (sanity check):   obs is flat vector, no encoder
  2. MSDP/Concat SAC:                  obs is dict; encoder is external and frozen;
                                        actor uses pooled embeddings, critic uses cross-attention.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

# ─── hyperparameters ────────────────────────────────────────────
GAMMA   = 0.99
TAU     = 0.005
LR      = 3e-4
ALPHA_INIT = 0.1   # initial entropy coefficient


# ═══════════════════════════════════════════════════════════════════
#  Replay Buffer
# ═══════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Stores transitions. For dict observations (multisensory), stores components separately.
    Images stored as uint8 to save memory.
    """
    def __init__(self, capacity: int, obs_spec: dict | int, action_dim: int, device: str = 'cpu'):
        """
        obs_spec: dict like {'proprio': (14,), 'ft': (4,6), 'vision': (64,64,3)}
                  or int for flat state.
        """
        self.capacity = capacity
        self.device = device
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self._dict_obs = isinstance(obs_spec, dict)
        if self._dict_obs:
            self.obs_bufs = {
                k: np.zeros((capacity, *shape),
                             dtype=np.uint8 if 'vision' in k else np.float32)
                for k, shape in obs_spec.items()
            }
            self.next_obs_bufs = {
                k: np.zeros((capacity, *shape),
                             dtype=np.uint8 if 'vision' in k else np.float32)
                for k, shape in obs_spec.items()
            }
        else:
            self.obs_bufs      = np.zeros((capacity, obs_spec), dtype=np.float32)
            self.next_obs_bufs = np.zeros((capacity, obs_spec), dtype=np.float32)

        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1), dtype=np.float32)
        self.dones       = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        if self._dict_obs:
            for k in self.obs_bufs:
                self.obs_bufs[k][self.ptr]      = obs[k]
                self.next_obs_bufs[k][self.ptr] = next_obs[k]
        else:
            self.obs_bufs[self.ptr]      = obs
            self.next_obs_bufs[self.ptr] = next_obs

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr]   = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        dev = self.device

        def to_tensor(arr, dtype=torch.float32):
            return torch.as_tensor(arr, dtype=dtype).to(dev)

        if self._dict_obs:
            obs = {k: to_tensor(self.obs_bufs[k][idx]) for k in self.obs_bufs}
            next_obs = {k: to_tensor(self.next_obs_bufs[k][idx]) for k in self.next_obs_bufs}
        else:
            obs      = to_tensor(self.obs_bufs[idx])
            next_obs = to_tensor(self.next_obs_bufs[idx])

        return {
            'obs':      obs,
            'action':   to_tensor(self.actions[idx]),
            'reward':   to_tensor(self.rewards[idx]),
            'next_obs': next_obs,
            'done':     to_tensor(self.dones[idx]),
        }

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════
#  Networks: State-based SAC (sanity check)
# ═══════════════════════════════════════════════════════════════════

def mlp(dims: list[int], activation=nn.GELU, output_activation=None) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        act = activation if i < len(dims) - 2 else output_activation
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class StateCritic(nn.Module):
    """Twin Q-network for state-based SAC."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp([state_dim + action_dim, hidden, hidden, 1])
        self.q2 = mlp([state_dim + action_dim, hidden, hidden, 1])

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class StateActor(nn.Module):
    """Gaussian actor with reparameterisation."""
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([state_dim, hidden, hidden])
        self.mu_head  = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = self.net(state)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mu, std)
        x_t     = dist.rsample()
        action  = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


# ═══════════════════════════════════════════════════════════════════
#  Networks: MSDP / Concat latent-based SAC (downstream RL)
# ═══════════════════════════════════════════════════════════════════

class CrossAttentionCritic(nn.Module):
    """
    MSDP critic: learnable query, K/V = frozen encoder embeddings.
    One cross-attention layer → twin Q heads.
    """
    def __init__(self, d_model: int = 128, action_dim: int = 3, n_heads: int = 4, n_tokens: int = 66):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # Twin Q heads
        self.q1 = mlp([d_model + action_dim, 256, 256, 1])
        self.q2 = mlp([d_model + action_dim, 256, 256, 1])

    def forward(self, embeddings: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        embeddings: (B, T, d_model) — frozen encoder output
        action:     (B, action_dim)
        Returns:    q1, q2 each (B, 1)
        """
        B = embeddings.shape[0]
        q = self.query.expand(B, -1, -1)                    # (B, 1, d)
        attn_out, _ = self.cross_attn(q, embeddings, embeddings)  # (B, 1, d)
        z = self.norm(attn_out.squeeze(1))                  # (B, d)
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)


class PooledActor(nn.Module):
    """
    MSDP actor: mean pool over vision tokens first, then mean over all tokens.
    Stop-gradient (frozen encoder embeddings passed in; no grad through encoder).
    """
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, d_model: int = 128, action_dim: int = 3,
                 n_vision_tokens: int = 64, vision_start: int = 2):
        super().__init__()
        self.n_vis = n_vision_tokens
        self.vis_start = vision_start
        self.net = mlp([d_model, 256, 256])
        self.mu_head      = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def pool(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, 66, d_model)
        Returns pooled: (B, d_model)
        """
        vis_mean = embeddings[:, self.vis_start: self.vis_start + self.n_vis].mean(dim=1)  # (B, d)
        low_dim  = embeddings[:, :self.vis_start]                                          # (B, 2, d)
        # Pool all: [low_dim_tokens..., vision_mean]
        all_toks = torch.cat([low_dim, vis_mean.unsqueeze(1)], dim=1)  # (B, 3, d)
        return all_toks.mean(dim=1)  # (B, d)

    def forward(self, embeddings: torch.Tensor):
        """
        embeddings: (B, 66, d_model) — stop-gradient already applied by caller
        """
        z = self.pool(embeddings.detach())  # stop-gradient
        h = self.net(z)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mu, std)
        x_t     = dist.rsample()
        action  = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class ConcatLatentCritic(nn.Module):
    """Twin Q for concat baseline (flat latent + action)."""
    def __init__(self, latent_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.q1 = mlp([latent_dim + action_dim, 256, 256, 1])
        self.q2 = mlp([latent_dim + action_dim, 256, 256, 1])

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)


class ConcatLatentActor(nn.Module):
    """Actor on top of concat latent (flat)."""
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, latent_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.net = mlp([latent_dim, 256, 256])
        self.mu_head      = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, z: torch.Tensor):
        h = self.net(z.detach())  # stop-gradient from encoder
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mu, std)
        x_t     = dist.rsample()
        action  = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


# ═══════════════════════════════════════════════════════════════════
#  Ablation: Pooled Critic (mean pooling for both actor & critic)
# ═══════════════════════════════════════════════════════════════════

class PooledCritic(nn.Module):
    """Ablation: mean pool for critic too (instead of cross-attention)."""
    def __init__(self, d_model: int = 128, action_dim: int = 3,
                 n_vision_tokens: int = 64, vision_start: int = 2):
        super().__init__()
        self.n_vis = n_vision_tokens
        self.vis_start = vision_start
        self.q1 = mlp([d_model + action_dim, 256, 256, 1])
        self.q2 = mlp([d_model + action_dim, 256, 256, 1])

    def pool(self, embeddings: torch.Tensor) -> torch.Tensor:
        vis_mean = embeddings[:, self.vis_start: self.vis_start + self.n_vis].mean(dim=1)
        low_dim  = embeddings[:, :self.vis_start]
        all_toks = torch.cat([low_dim, vis_mean.unsqueeze(1)], dim=1)
        return all_toks.mean(dim=1)

    def forward(self, embeddings: torch.Tensor, action: torch.Tensor):
        z = self.pool(embeddings)
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)


class CLSCritic(nn.Module):
    """Ablation: CLS token for critic. We use token 0 as CLS (proprio position used as CLS)."""
    def __init__(self, d_model: int = 128, action_dim: int = 3):
        super().__init__()
        self.q1 = mlp([d_model + action_dim, 256, 256, 1])
        self.q2 = mlp([d_model + action_dim, 256, 256, 1])

    def forward(self, embeddings: torch.Tensor, action: torch.Tensor):
        # Token 0 = proprio token acts as CLS
        z = embeddings[:, 0]
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)


# ═══════════════════════════════════════════════════════════════════
#  SAC Agent
# ═══════════════════════════════════════════════════════════════════

class SACAgent:
    """
    Soft Actor-Critic.
    mode: 'state' | 'msdp' | 'concat' | 'ablation_pool' | 'ablation_cls'
    """

    def __init__(
        self,
        obs_spec,
        action_dim: int,
        device: str = 'cuda',
        mode: str = 'state',
        encoder=None,        # pretrained encoder (frozen); None for state mode
        d_model: int = 128,
        hidden_state: int = 256,
        lr: float = LR,
        gamma: float = GAMMA,
        tau: float = TAU,
        alpha_init: float = ALPHA_INIT,
        target_entropy: float | None = None,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
    ):
        self.device   = torch.device(device)
        self.mode     = mode
        self.gamma    = gamma
        self.tau      = tau
        self.batch_sz = batch_size
        self.action_dim = action_dim

        # Frozen encoder
        self.encoder = encoder
        if encoder is not None:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # Build networks
        if mode == 'state':
            state_dim = obs_spec if isinstance(obs_spec, int) else np.prod(obs_spec)
            self.actor   = StateActor(state_dim, action_dim, hidden_state).to(self.device)
            self.critic  = StateCritic(state_dim, action_dim, hidden_state).to(self.device)
            self.critic_target = deepcopy(self.critic).to(self.device)

        elif mode in ('msdp', 'msdp_r'):
            self.actor  = PooledActor(d_model, action_dim).to(self.device)
            self.critic = CrossAttentionCritic(d_model, action_dim).to(self.device)
            self.critic_target = CrossAttentionCritic(d_model, action_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        elif mode == 'concat':
            self.actor  = ConcatLatentActor(d_model, action_dim).to(self.device)
            self.critic = ConcatLatentCritic(d_model, action_dim).to(self.device)
            self.critic_target = ConcatLatentCritic(d_model, action_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        elif mode == 'ablation_pool':
            self.actor  = PooledActor(d_model, action_dim).to(self.device)
            self.critic = PooledCritic(d_model, action_dim).to(self.device)
            self.critic_target = PooledCritic(d_model, action_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        elif mode == 'ablation_cls':
            self.actor  = PooledActor(d_model, action_dim).to(self.device)
            self.critic = CLSCritic(d_model, action_dim).to(self.device)
            self.critic_target = CLSCritic(d_model, action_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Target entropy: -|A|
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy

        # Auto-entropy tuning
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32,
                                       requires_grad=True, device=self.device)
        self.alpha = alpha_init

        # Optimisers
        actor_params  = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        self.actor_opt  = torch.optim.Adam(actor_params,  lr=lr)
        self.critic_opt = torch.optim.Adam(critic_params, lr=lr)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, obs_spec, action_dim, str(self.device))

    @torch.no_grad()
    def _encode(self, obs) -> torch.Tensor:
        """Run frozen encoder on obs dict; return embeddings."""
        if self.encoder is None:
            # State mode: obs is already a tensor
            return obs
        if self.mode == 'concat':
            return self.encoder(obs)          # (B, d_model)
        else:
            return self.encoder(obs)          # (B, 66, d_model)

    def select_action(self, obs, deterministic: bool = False) -> np.ndarray:
        """Select action for a single obs (unbatched)."""
        # Move to device
        if isinstance(obs, dict):
            obs_t = {k: torch.as_tensor(v, dtype=torch.float32 if 'vision' not in k else torch.uint8)
                     .unsqueeze(0).to(self.device) for k, v in obs.items()}
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.mode == 'state':
                _, _, mu = self.actor(obs_t)
                action, _, _ = self.actor(obs_t)
            else:
                emb = self._encode(obs_t)
                if self.mode == 'concat':
                    action, _, mu = self.actor(emb)
                else:
                    action, _, mu = self.actor(emb)

        result = (mu if deterministic else action)
        return result.squeeze(0).cpu().numpy()

    def update(self) -> dict | None:
        """One SAC gradient step. Returns metrics dict or None if buffer too small."""
        if len(self.buffer) < self.batch_sz:
            return None

        batch = self.buffer.sample(self.batch_sz)
        obs   = batch['obs']
        nobs  = batch['next_obs']
        act   = batch['action']
        rew   = batch['reward']
        done  = batch['done']

        # ── Critic update ──
        with torch.no_grad():
            if self.mode == 'state':
                next_act, next_lp, _ = self.actor(nobs)
                q1_t, q2_t = self.critic_target(nobs, next_act)
            elif self.mode == 'concat':
                n_emb = self._encode(nobs)
                next_act, next_lp, _ = self.actor(n_emb)
                q1_t, q2_t = self.critic_target(n_emb, next_act)
            else:
                n_emb = self._encode(nobs)
                next_act, next_lp, _ = self.actor(n_emb)
                q1_t, q2_t = self.critic_target(n_emb, next_act)

            min_q_t = torch.min(q1_t, q2_t) - self.alpha * next_lp
            q_target = rew + self.gamma * (1 - done) * min_q_t

        if self.mode == 'state':
            q1, q2 = self.critic(obs, act)
        elif self.mode == 'concat':
            emb = self._encode(obs)
            q1, q2 = self.critic(emb, act)
        else:
            emb = self._encode(obs)
            q1, q2 = self.critic(emb, act)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ──
        if self.mode == 'state':
            new_act, lp, _ = self.actor(obs)
            q1_a, q2_a = self.critic(obs, new_act)
        elif self.mode == 'concat':
            emb_a = self._encode(obs)
            new_act, lp, _ = self.actor(emb_a)
            q1_a, q2_a = self.critic(emb_a, new_act)
        else:
            emb_a = self._encode(obs)
            new_act, lp, _ = self.actor(emb_a)
            q1_a, q2_a = self.critic(emb_a, new_act)

        min_q_a = torch.min(q1_a, q2_a)
        actor_loss = (self.alpha * lp - min_q_a).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Alpha (entropy) update ──
        alpha_loss = -(self.log_alpha.exp() * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # ── Soft target update ──
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss':  actor_loss.item(),
            'alpha':       self.alpha,
            'alpha_loss':  alpha_loss.item(),
        }

    def save(self, path: str):
        torch.save({
            'actor':  self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.log_alpha.data = ckpt['log_alpha'].data
        self.alpha = self.log_alpha.exp().item()
        self.critic_target.load_state_dict(self.critic.state_dict())
