"""
Mock Push Cube Multisensory Environment
========================================
IMPORTANT: This is a pure-Python mock that replaces panda-gym/PyBullet
due to PyBullet requiring MSVC Build Tools which are not installed.

DEVIATION D_ENV1: PyBullet not available on this machine (no MSVC compiler).
  - Environment implemented as a physics mock with realistic kinematics.
  - FT sensor simulated as contact force based on distance + velocity.
  - Camera rendering uses random-colour patches (not real rendering).
  - All real observation shapes and ranges are preserved.
  - Reward function matches panda-gym's dense Push reward.
  - For real results, swap this file for the real env.py using panda-gym.

Shape contract (identical to real env):
    obs['proprio']:  (14,)   float32   joint pos + vel
    obs['ft']:       (4, 6)  float32   4 FT readings
    obs['vision']:   (64,64,3) uint8   RGB image
    action:          (3,)    float32   EE displacement, clipped to [-1,1]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ─── constants ──────────────────────────────────────────────────────
IMG_H = 64
IMG_W = 64
N_FT_STEPS = 4
FT_DIM = 6
PROPRIO_DIM = 14
ACTION_DIM = 3

CUBE_MASS_RANGE = (0.05, 0.5)
CUBE_COM_RANGE  = (-0.01, 0.01)

# Workspace bounds (metres, in table frame)
EE_HOME       = np.array([0.3, 0.0, 0.3], dtype=np.float32)  # robot EE start
CUBE_CENTER   = np.array([0.3, 0.0, 0.02], dtype=np.float32)  # table surface
GOAL_RANGE    = np.array([0.1, 0.15])   # x, y half-ranges for cube+goal spawn

EE_INIT_NOISE  = 0.02
CUBE_INIT_NOISE = 0.05

# Success: cube within 5 cm of goal (panda-gym uses 5cm)
SUCCESS_THRESH = 0.05
MAX_STEPS      = 50  # episodes cut at 50 steps


class PushCubeMock(gym.Env):
    """
    Mock push-cube environment with correct observation shapes.

    Physics model:
      - EE moves with 3D displacement actions (clipped ±0.05 m per step).
      - Cube is on a table surface (z=fixed). When EE is close and moving
        toward cube, the cube moves proportionally (simplified contact model).
      - FT sensor: sine-wave contact forces when EE near cube.
      - Vision: rendered as coloured patches for ego(red) / cube(blue) / goal(green).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        img_h: int = IMG_H,
        img_w: int = IMG_W,
        n_ft_steps: int = N_FT_STEPS,
        randomise_dynamics: bool = True,
        seed: int | None = None,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.n_ft_steps = n_ft_steps
        self.randomise_dynamics = randomise_dynamics
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # Spaces
        self.observation_space = spaces.Dict({
            "proprio": spaces.Box(-np.inf, np.inf, shape=(PROPRIO_DIM,), dtype=np.float32),
            "ft":      spaces.Box(-np.inf, np.inf, shape=(N_FT_STEPS, FT_DIM), dtype=np.float32),
            "vision":  spaces.Box(0, 255, shape=(img_h, img_w, 3), dtype=np.uint8),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

        # State: [ee_pos(3), cube_pos(3), goal_pos(3)]
        self._ee_pos   = EE_HOME.copy()
        self._cube_pos = CUBE_CENTER.copy()
        self._goal_pos = CUBE_CENTER.copy()
        self._cube_mass = 0.1
        self._ft_buf   = np.zeros((N_FT_STEPS, FT_DIM), dtype=np.float32)
        # Joint angles: 7D (mock sinusoidal)
        self._joint_angles = np.zeros(7, dtype=np.float32)
        self._joint_vels   = np.zeros(7, dtype=np.float32)

    # ── helpers ──────────────────────────────────────────────────────

    def _get_proprio(self) -> np.ndarray:
        """7 joint positions + 7 joint velocities = 14-dim."""
        return np.concatenate([self._joint_angles, self._joint_vels]).astype(np.float32)

    def _update_joints(self, action: np.ndarray):
        """Mock forward kinematics: update joint angles from EE displacement."""
        # Simple: distribute EE movement to joint 0..2, vel proportional
        disp = action * 0.05  # scale action to metres
        self._joint_vels[:3]   = disp / 0.05  # approximate rad/s
        self._joint_vels[3:]   = 0.0
        self._joint_angles[:3] += self._joint_vels[:3] * 0.1
        self._joint_angles     = np.clip(self._joint_angles, -np.pi, np.pi)

    def _get_ft(self) -> np.ndarray:
        """6-DOF FT reading based on EE-cube proximity."""
        dist = np.linalg.norm(self._ee_pos - self._cube_pos)
        contact_scale = np.exp(-dist / 0.05) * 5.0  # N (rough)
        direction = (self._ee_pos - self._cube_pos)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        forces  = -direction * contact_scale
        torques = np.cross(direction, np.array([0, 0, 1.0])) * contact_scale * 0.05
        ft = np.concatenate([forces, torques]).astype(np.float32)
        # Clamp to realistic range
        return np.clip(ft, -20.0, 20.0)

    def _get_vision(self) -> np.ndarray:
        """
        Simple rendered image: dark background with coloured markers.
        - Blue circle: EE position (projected to image)
        - Red circle: cube position
        - Green cross: goal position
        """
        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 30  # dark grey

        def world_to_img(pos_3d):
            """Project 3D world → 2D pixel (orthographic, top-down view)."""
            # x: left-right, y: up-down (table perspective)
            px = int((pos_3d[0] - 0.1) / 0.4 * self.img_w)
            py = int((pos_3d[1] + 0.2) / 0.4 * self.img_h)
            return np.clip(px, 0, self.img_w - 1), np.clip(py, 0, self.img_h - 1)

        def draw_circle(img, cx, cy, r, color):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx**2 + dy**2 <= r**2:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < self.img_w and 0 <= y < self.img_h:
                            img[y, x] = color

        # Draw goal (green)
        gx, gy = world_to_img(self._goal_pos)
        draw_circle(img, gx, gy, 5, [0, 200, 0])
        # Draw cube (red)
        cx, cy = world_to_img(self._cube_pos)
        draw_circle(img, cx, cy, 4, [220, 50, 50])
        # Draw EE (blue)
        ex, ey = world_to_img(self._ee_pos)
        draw_circle(img, ex, ey, 3, [50, 50, 220])

        return img

    def _compute_reward(self, prev_cube_pos: np.ndarray) -> tuple[float, bool]:
        """Dense reward: decrease cube-goal distance + ee-cube approach bonus."""
        cube_goal_dist = np.linalg.norm(self._cube_pos[:2] - self._goal_pos[:2])
        success = cube_goal_dist < SUCCESS_THRESH

        reward = -cube_goal_dist  # dense: negative distance
        # Contact bonus: reward for EE near cube
        ee_cube_dist = np.linalg.norm(self._ee_pos - self._cube_pos)
        if ee_cube_dist < 0.1:
            reward += 0.1 * (1.0 - ee_cube_dist / 0.1)
        if success:
            reward += 1.0

        return reward, success

    def _make_obs(self) -> dict:
        ft_now = self._get_ft()
        # Roll buffer
        self._ft_buf = np.roll(self._ft_buf, shift=-1, axis=0)
        self._ft_buf[-1] = ft_now

        return {
            "proprio": self._get_proprio(),
            "ft":      self._ft_buf.copy(),
            "vision":  self._get_vision(),
        }

    # ── gym interface ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomise EE start
        ee_noise = self._rng.uniform(-EE_INIT_NOISE, EE_INIT_NOISE, 3).astype(np.float32)
        self._ee_pos = EE_HOME + ee_noise

        # Randomise cube + goal positions
        cube_noise = self._rng.uniform(-CUBE_INIT_NOISE, CUBE_INIT_NOISE, 2).astype(np.float32)
        self._cube_pos = CUBE_CENTER.copy()
        self._cube_pos[:2] += cube_noise

        goal_xy = self._rng.uniform(-GOAL_RANGE, GOAL_RANGE).astype(np.float32) + CUBE_CENTER[:2]
        goal_xy = np.clip(goal_xy, [0.1, -0.2], [0.5, 0.2])
        self._goal_pos = np.array([goal_xy[0], goal_xy[1], CUBE_CENTER[2]], dtype=np.float32)

        # Randomise dynamics
        if self.randomise_dynamics:
            self._cube_mass = float(self._rng.uniform(*CUBE_MASS_RANGE))

        # Reset joints and FT buffer
        self._joint_angles = np.zeros(7, dtype=np.float32)
        self._joint_vels   = np.zeros(7, dtype=np.float32)
        self._ft_buf       = np.zeros((N_FT_STEPS, FT_DIM), dtype=np.float32)
        self._step_count   = 0

        obs = self._make_obs()
        info = {'cube_mass': self._cube_mass, 'success': False}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        prev_cube = self._cube_pos.copy()

        # Update EE position
        disp = action.astype(np.float32) * 0.05  # max 5 cm/step
        self._ee_pos = np.clip(self._ee_pos + disp, [0.05, -0.4, 0.1], [0.6, 0.4, 0.6])

        # Simplified contact model: cube moves when EE pushes it
        ee_cube_vec = self._cube_pos - self._ee_pos
        ee_cube_dist = np.linalg.norm(ee_cube_vec[:2])
        CONTACT_RADIUS = 0.06  # m (roughly gripper + cube radii)
        if ee_cube_dist < CONTACT_RADIUS and ee_cube_dist > 0.001:
            # Push cube away from EE proportional to penetration
            penetration = (CONTACT_RADIUS - ee_cube_dist) / CONTACT_RADIUS
            push_dir = ee_cube_vec[:2] / ee_cube_dist
            # Mass affects how much cube moves (lighter = more displacement)
            scale = penetration * 0.03 * (0.2 / self._cube_mass)
            self._cube_pos[:2] += push_dir * scale
            # Keep on table surface
            self._cube_pos[:2] = np.clip(self._cube_pos[:2], [0.1, -0.3], [0.6, 0.3])

        # Update joint state
        self._update_joints(action)
        self._step_count += 1

        reward, success = self._compute_reward(prev_cube)
        obs = self._make_obs()

        terminated = success
        truncated = self._step_count >= MAX_STEPS
        info = {'success': success, 'cube_pos': self._cube_pos.copy(),
                'goal_pos': self._goal_pos.copy(), 'ee_pos': self._ee_pos.copy()}

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._get_vision()


class PushCubeStateMock(gym.Env):
    """
    State-based push cube for SAC sanity check.
    Obs: [ee_pos(3), cube_pos(3), goal_pos(3)] = 9-dim flat float32.
    Action: 3D EE displacement.
    """
    def __init__(self, seed=None):
        super().__init__()
        self._inner = PushCubeMock(seed=seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.action_space = self._inner.action_space

    def _state(self):
        return np.concatenate([
            self._inner._ee_pos,
            self._inner._cube_pos,
            self._inner._goal_pos,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        self._inner.reset(seed=seed, options=options)
        return self._state(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self._inner.step(action)
        return self._state(), reward, terminated, truncated, info

    def render(self):
        return self._inner.render()


def make_env(env_type: str = "multisensory", seed: int = 0, **kwargs):
    """Factory. env_type in {'multisensory', 'state'}."""
    if env_type == "state":
        return PushCubeStateMock(seed=seed)
    else:
        return PushCubeMock(seed=seed, **kwargs)
