"""
Environment wrapper and preprocessing utilities for the Mario World Model.

This module defines a wrapper around the gym-super-mario-bros environment
that performs preprocessing such as grayscale conversion, resizing and
frame stacking.  It also exposes the discrete action set used by the agent.
"""

import numpy as np
import cv2

try:
    # gym-super-mario-bros and nes-py are required for the Mario environment
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    # Right-only action set: NOOP, right, right+jump
    from gym_super_mario_bros.actions import RIGHT_ONLY
except ImportError:
    gym_super_mario_bros = None
    JoypadSpace = None
    RIGHT_ONLY = None


class MarioEnvWrapper:
    """
    Wraps the Super Mario Bros environment to provide preprocessing and
    a consistent observation shape.  Observations are converted to
    grayscale, resized and normalised to [0,1] and stacked along the
    channel dimension.
    """

    def __init__(self,
                 env_name: str = "SuperMarioBros-1-1-v0",
                 frame_stack: int = 4,
                 resize: tuple = (64, 64)):
        if gym_super_mario_bros is None or JoypadSpace is None:
            raise ImportError(
                "gym-super-mario-bros and nes-py are required for the Mario environment. "
                "Install them via pip install gym-super-mario-bros nes-py."
            )
        # Create base environment and restrict to right-only actions
        env = gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, RIGHT_ONLY)
        self.env = env
        self.frame_stack = frame_stack
        self.resize = resize
        self.frames = []
        # Define observation and action spaces for compatibility with gym
        from gym import spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(frame_stack, resize[0], resize[1]),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB frame to grayscale, resize and normalise to [0,1]."""
        # Convert from HWC RGB to single-channel grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to target resolution
        resized = cv2.resize(gray, self.resize, interpolation=cv2.INTER_AREA)
        # Normalise pixel values
        normalised = resized.astype(np.float32) / 255.0
        return normalised

    def reset(self) -> np.ndarray:
        """Reset the underlying environment and return the initial stacked observation."""
        self.frames = []
        obs = self.env.reset()
        processed = self._preprocess(obs)
        # Start with repeated initial frame
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)

    def step(self, action: int):
        """
        Apply the given action in the environment and return:
        stacked_obs, reward, done, info
        """
        obs, reward, done, info = self.env.step(action)
        processed = self._preprocess(obs)
        # Maintain a rolling buffer of the last frame_stack frames
        self.frames.pop(0)
        self.frames.append(processed)
        stacked = np.stack(self.frames, axis=0)
        return stacked, reward, done, info

    def close(self):
        """Close the underlying environment."""
        self.env.close()