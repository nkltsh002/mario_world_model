"""Environment wrapper and preprocessing utilities for the Mario World Model."""

from __future__ import annotations

import os
import pickle
import random
from typing import List

import numpy as np
from gym import spaces

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - executed only in headless envs
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False
    from skimage.color import rgb2gray
    from skimage.transform import resize

try:
    # gym-super-mario-bros and nes-py are required for the Mario environment
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    # Right-only action set: NOOP, right, right+jump
    from gym_super_mario_bros.actions import RIGHT_ONLY
except ImportError:  # pragma: no cover - dependency not installed
    gym_super_mario_bros = None
    JoypadSpace = None
    RIGHT_ONLY = None


class _OfflineMarioEnv:
    """Lightweight fallback environment that replays recorded trajectories."""

    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
        if len(trajectories) == 0:
            raise RuntimeError("Offline dataset is empty; cannot create fallback environment.")
        self.trajectories: List[dict] = trajectories
        # Infer dimensions from dataset
        example = trajectories[0]["observations"]
        self.frame_stack = example.shape[1]
        self.resize = (example.shape[2], example.shape[3])
        max_action = 0
        for traj in trajectories:
            max_action = max(max_action, int(traj["actions"].max()))
        self.action_space = spaces.Discrete(max_action + 1)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.frame_stack, *self.resize),
            dtype=np.float32,
        )
        self._current = None
        self._step = 0

    def reset(self):
        self._current = random.choice(self.trajectories)
        self._step = 0
        return self._current["observations"][0].astype(np.float32)

    def step(self, action: int):  # pylint: disable=unused-argument
        assert self._current is not None, "Environment must be reset before stepping."
        self._step += 1
        observations = self._current["observations"]
        rewards = self._current["rewards"]
        done = self._step >= len(observations) - 1
        obs_index = min(self._step, len(observations) - 1)
        reward_index = min(self._step - 1, len(rewards) - 1)
        reward = float(rewards[reward_index]) if self._step > 0 else 0.0
        next_obs = observations[obs_index].astype(np.float32)
        return next_obs, reward, done, {"offline": True}

    def close(self):
        self._current = None


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
        self.frame_stack = frame_stack
        self.resize = resize
        self.frames = []
        self._offline = False

        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "mario_rollouts_synth.pkl"
        )

        try:
            if gym_super_mario_bros is None or JoypadSpace is None:
                raise ImportError("gym-super-mario-bros not available")
            env = gym_super_mario_bros.make(env_name)
            env = JoypadSpace(env, RIGHT_ONLY)
            self.env = env
            self.action_space = env.action_space
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(frame_stack, resize[0], resize[1]),
                dtype=np.float32,
            )
        except Exception as exc:  # pragma: no cover - exercised in headless CI
            if not os.path.exists(dataset_path):
                raise RuntimeError(
                    "Failed to initialise Super Mario Bros environment and no offline dataset found."
                ) from exc
            self._offline = True
            self.env = _OfflineMarioEnv(dataset_path)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            self.frame_stack = self.env.frame_stack
            self.resize = self.env.resize

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB frame to grayscale, resize and normalise to [0,1]."""
        if self._offline:
            return frame.astype(np.float32)
        if _CV2_AVAILABLE:
            # Convert from HWC RGB to single-channel grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # cv2 expects the (width, height) ordering
            resized = cv2.resize(gray, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
            # Normalise pixel values
            normalised = resized.astype(np.float32) / 255.0
            return normalised

        # Fall back to scikit-image for environments where OpenCV is not
        # available (e.g. headless CI containers without libGL).
        gray = rgb2gray(frame).astype(np.float32)
        resized = resize(gray, self.resize, anti_aliasing=True).astype(np.float32)
        return resized

    def reset(self) -> np.ndarray:
        """Reset the underlying environment and return the initial stacked observation."""
        if self._offline:
            return self.env.reset()

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
        if self._offline:
            return self.env.step(action)

        obs, reward, done, info = self.env.step(action)
        processed = self._preprocess(obs)
        # Maintain a rolling buffer of the last frame_stack frames
        self.frames.pop(0)
        self.frames.append(processed)
        stacked = np.stack(self.frames, axis=0)
        return stacked, reward, done, info

    def close(self):
        """Close the underlying environment."""
        if hasattr(self.env, "close"):
            self.env.close()
