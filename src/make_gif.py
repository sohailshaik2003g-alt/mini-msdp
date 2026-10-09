"""
GIF generation: random policy episode.
Run: python src/make_gif.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import imageio.v2 as imageio
from env import make_env

def make_random_gif(path: str = 'media/random_policy.gif', n_steps: int = 60, seed: int = 0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    env = make_env('multisensory', seed=seed)
    obs, _ = env.reset(seed=seed)
    frames = []

    for step in range(n_steps):
        frame = env.render()  # (64, 64, 3) uint8
        # Upscale for visibility
        import numpy as np
        frame_up = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)  # 256x256
        frames.append(frame_up)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    imageio.mimsave(path, frames, fps=10, loop=0)
    print(f"Saved GIF: {path} ({n_steps} frames)")


if __name__ == '__main__':
    make_random_gif()
