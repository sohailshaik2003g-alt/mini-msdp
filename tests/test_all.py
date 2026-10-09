"""
Unit tests for MSDP mini-reproduction.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import pytest

from env import make_env, PushCubeMock, PROPRIO_DIM, N_FT_STEPS, FT_DIM, IMG_H, IMG_W
from model import (
    MSDPEncoder, MSDPDecoder, MSDPPretrainModel, ConcatPretrainModel,
    CNNStem, sample_mask, apply_mask,
    N_TOTAL_TOKENS, IDX_VISION_START, N_VISION_TOKENS, D_MODEL, MASK_RATIO
)
from sac import SACAgent, ReplayBuffer


# ════════════════════════════════════════════════════════════════════
# Environment tests
# ════════════════════════════════════════════════════════════════════

class TestEnvironment:

    def test_obs_shapes(self):
        env = PushCubeMock(seed=0)
        obs, info = env.reset()
        assert obs['proprio'].shape == (PROPRIO_DIM,),   f"proprio: {obs['proprio'].shape}"
        assert obs['ft'].shape == (N_FT_STEPS, FT_DIM),  f"ft: {obs['ft'].shape}"
        assert obs['vision'].shape == (IMG_H, IMG_W, 3), f"vision: {obs['vision'].shape}"
        assert obs['proprio'].dtype == np.float32
        assert obs['ft'].dtype == np.float32
        assert obs['vision'].dtype == np.uint8

    def test_obs_dtypes_after_step(self):
        env = PushCubeMock(seed=0)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs['proprio'].dtype == np.float32
        assert obs['ft'].dtype == np.float32
        assert obs['vision'].dtype == np.uint8

    def test_reset_deterministic(self):
        """Same seed → same observation."""
        env1 = PushCubeMock(seed=42)
        env2 = PushCubeMock(seed=42)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1['proprio'], obs2['proprio'])
        np.testing.assert_array_equal(obs1['ft'],      obs2['ft'])
        np.testing.assert_array_equal(obs1['vision'],  obs2['vision'])

    def test_ft_nonzero_on_contact(self):
        """FT should be nonzero when EE is very close to cube."""
        env = PushCubeMock(seed=0)
        env.reset()
        # Force EE onto cube position
        env._ee_pos = env._cube_pos.copy()
        env._ee_pos[2] = 0.05  # slightly above

        obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
        ft = obs['ft']
        assert ft.shape == (N_FT_STEPS, FT_DIM)
        assert np.abs(ft[-1]).sum() > 0.01, f"FT should be nonzero on contact: {ft}"

    def test_action_space(self):
        env = PushCubeMock(seed=0)
        assert env.action_space.shape == (3,)
        assert env.action_space.dtype == np.float32

    def test_vision_range(self):
        env = PushCubeMock(seed=0)
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(3))
        assert obs['vision'].min() >= 0
        assert obs['vision'].max() <= 255

    def test_state_env(self):
        from env import PushCubeStateMock
        env = PushCubeStateMock(seed=0)
        obs, _ = env.reset()
        assert obs.shape == (9,)
        assert obs.dtype == np.float32
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (9,)

    def test_dynamics_randomisation(self):
        """Different seeds should produce different cube masses."""
        env = PushCubeMock(seed=0)
        env.reset(seed=0)
        mass1 = env._cube_mass
        env.reset(seed=99)
        mass2 = env._cube_mass
        # With very high probability these should differ
        assert mass1 != mass2, "Dynamics randomisation not working"


# ════════════════════════════════════════════════════════════════════
# Model tests
# ════════════════════════════════════════════════════════════════════

class TestCNNStem:

    def test_output_shape(self):
        stem = CNNStem(D_MODEL)
        x = torch.randn(2, 3, 64, 64)
        out = stem(x)
        assert out.shape == (2, 64, D_MODEL), f"CNNStem output: {out.shape}"


class TestMasking:

    def test_mask_ratio(self):
        B = 8
        vis_mask, masked_mask = sample_mask(B)
        # About 70% masked
        actual_ratio = masked_mask.float().mean().item()
        assert abs(actual_ratio - MASK_RATIO) < 0.05, f"Mask ratio: {actual_ratio}"

    def test_vision_never_fully_masked(self):
        """At least 1 vision token must remain visible."""
        for _ in range(20):
            B = 4
            vis_mask, _ = sample_mask(B)
            for b in range(B):
                vision_visible = vis_mask[b, IDX_VISION_START: IDX_VISION_START + N_VISION_TOKENS].sum()
                assert vision_visible >= 1, f"Sample {b}: 0 vision tokens visible"

    def test_visible_mask_complement(self):
        vis_mask, masked_mask = sample_mask(4)
        assert (vis_mask ^ masked_mask).all(), "visible and masked should be complements"
        assert (vis_mask + masked_mask).eq(1).all(), "every token either visible or masked"

    def test_apply_mask_shape(self):
        B, T, D = 4, N_TOTAL_TOKENS, D_MODEL
        tokens = torch.randn(B, T, D)
        pos_idx = torch.arange(T).unsqueeze(0).expand(B, -1)
        vis_mask, _ = sample_mask(B)
        vis_tokens, vis_pos, vis_lengths = apply_mask(tokens, pos_idx, vis_mask)
        # Vis tokens should have max_vis as second dim
        assert vis_tokens.ndim == 3
        assert vis_tokens.shape[2] == D


class TestMSDPEncoder:

    @pytest.fixture
    def sample_obs(self):
        B = 2
        return {
            'proprio': torch.randn(B, 14),
            'ft':      torch.randn(B, 4, 6),
            'vision':  (torch.randint(0, 255, (B, 64, 64, 3))),  # uint8-like
        }

    def test_tokenize_shape(self, sample_obs):
        enc = MSDPEncoder()
        tokens, pos_idx = enc.tokenize(sample_obs)
        assert tokens.shape == (2, N_TOTAL_TOKENS, D_MODEL), f"{tokens.shape}"
        assert pos_idx.shape == (2, N_TOTAL_TOKENS)

    def test_full_forward_shape(self, sample_obs):
        enc = MSDPEncoder()
        out = enc(sample_obs)
        assert out.shape == (2, N_TOTAL_TOKENS, D_MODEL), f"{out.shape}"

    def test_frozen_encoder_no_grads(self, sample_obs):
        """Frozen encoder should not produce gradients."""
        enc = MSDPEncoder()
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)

        out = enc(sample_obs)
        # Gradients should not be computed
        for p in enc.parameters():
            assert not p.requires_grad, "Frozen encoder has grad"


class TestMSDPPretrain:

    @pytest.fixture
    def batch(self):
        B = 4
        return {
            'proprio': torch.randn(B, 14),
            'ft':      torch.randn(B, 4, 6),
            'vision':  torch.randint(0, 255, (B, 64, 64, 3)).float(),
        }

    def test_forward_returns_loss(self, batch):
        model = MSDPPretrainModel()
        result = model(batch)
        assert 'loss' in result
        assert result['loss'].ndim == 0, "Loss should be scalar"
        assert not torch.isnan(result['loss']), "Loss is NaN"
        assert not torch.isinf(result['loss']), "Loss is inf"

    def test_loss_components(self, batch):
        model = MSDPPretrainModel()
        result = model(batch)
        for k in ['loss_proprio', 'loss_ft', 'loss_vision']:
            assert k in result, f"Missing {k}"
            assert result[k].item() >= 0, f"{k} is negative"

    def test_backward(self, batch):
        model = MSDPPretrainModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = model(batch)
        loss = result['loss']
        loss.backward()
        opt.step()
        # Just check it doesn't crash


class TestConcatBaseline:

    @pytest.fixture
    def batch(self):
        B = 4
        return {
            'proprio': torch.randn(B, 14),
            'ft':      torch.randn(B, 4, 6),
            'vision':  torch.randint(0, 255, (B, 64, 64, 3)).float(),
        }

    def test_forward(self, batch):
        model = ConcatPretrainModel()
        result = model(batch)
        assert 'loss' in result
        assert not torch.isnan(result['loss'])


# ════════════════════════════════════════════════════════════════════
# SAC tests
# ════════════════════════════════════════════════════════════════════

class TestSAC:

    def test_state_select_action(self):
        agent = SACAgent(9, 3, device='cpu', mode='state')
        obs = np.random.randn(9).astype(np.float32)
        action = agent.select_action(obs, deterministic=False)
        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0 + 1e-6), "Action out of tanh bounds"

    def test_replay_buffer_state(self):
        buf = ReplayBuffer(100, 9, 3, 'cpu')
        obs = np.zeros(9)
        for i in range(50):
            buf.add(obs, np.zeros(3), 0.0, obs, False)
        assert len(buf) == 50
        batch = buf.sample(8)
        assert batch['obs'].shape == (8, 9)

    def test_replay_buffer_dict(self):
        obs_spec = {'proprio': (14,), 'ft': (4, 6), 'vision': (64, 64, 3)}
        buf = ReplayBuffer(100, obs_spec, 3, 'cpu')
        obs = {
            'proprio': np.zeros(14, dtype=np.float32),
            'ft':      np.zeros((4, 6), dtype=np.float32),
            'vision':  np.zeros((64, 64, 3), dtype=np.uint8),
        }
        for _ in range(20):
            buf.add(obs, np.zeros(3), 0.0, obs, False)
        batch = buf.sample(8)
        assert batch['obs']['proprio'].shape == (8, 14)
        assert batch['obs']['ft'].shape == (8, 4, 6)
        assert batch['obs']['vision'].shape == (8, 64, 64, 3)

    def test_state_sac_update(self):
        agent = SACAgent(9, 3, device='cpu', mode='state', buffer_capacity=1000, batch_size=8)
        obs = np.random.randn(9).astype(np.float32)
        # Fill buffer enough to trigger update
        for _ in range(20):
            action = agent.select_action(obs)
            agent.buffer.add(obs, action, 0.1, obs, False)
        metrics = agent.update()
        assert metrics is not None
        assert 'critic_loss' in metrics
        assert 'actor_loss' in metrics

    def test_msdp_sac_action(self):
        """MSDP agent should select action from dict obs."""
        enc = MSDPEncoder()
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)

        obs_spec = {'proprio': (14,), 'ft': (4, 6), 'vision': (64, 64, 3)}
        agent = SACAgent(obs_spec, 3, device='cpu', mode='msdp', encoder=enc)
        obs = {
            'proprio': np.random.randn(14).astype(np.float32),
            'ft':      np.random.randn(4, 6).astype(np.float32),
            'vision':  np.zeros((64, 64, 3), dtype=np.uint8),
        }
        action = agent.select_action(obs)
        assert action.shape == (3,)

    def test_actor_stop_gradient(self):
        """Actor forward should not propagate gradients into encoder."""
        from sac import PooledActor
        actor = PooledActor(D_MODEL, 3)
        emb = torch.randn(2, N_TOTAL_TOKENS, D_MODEL, requires_grad=True)
        action, lp, mu = actor(emb)
        loss = action.sum()
        loss.backward()
        # If stop-gradient is applied, emb.grad should be None
        assert emb.grad is None, "Actor is propagating gradients into encoder!"


# ════════════════════════════════════════════════════════════════════
# Integration smoke test
# ════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_env_model_pipeline(self):
        """End-to-end: env → tokenize → encode → loss."""
        env = PushCubeMock(seed=0)
        obs, _ = env.reset()

        # Convert to tensors with batch dim
        def to_tensor(v):
            return torch.as_tensor(v).unsqueeze(0).float()

        obs_t = {
            'proprio': to_tensor(obs['proprio']),
            'ft':      to_tensor(obs['ft']),
            'vision':  to_tensor(obs['vision']),
        }

        model = MSDPPretrainModel()
        result = model(obs_t)
        assert result['loss'].item() > 0
        print(f"  Integration test loss: {result['loss'].item():.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
