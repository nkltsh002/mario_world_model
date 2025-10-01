"""
Train a PPO agent on the Super Mario Bros environment for comparison.

This script uses stable‑baselines3's PPO implementation with an MLP policy.
Observations from the Mario environment are flattened and fed into the
policy network.  You may need to adjust hyper‑parameters for good
performance.

Usage:
    python -m mario_world_model.training.ppo_baseline --env SuperMarioBros-1-1-v0 \
        --timesteps 1000000 --output checkpoints/ppo_mario
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from utils.environment import MarioEnvWrapper
from config import MarioConfig


def make_mario_env(env_name: str):
    """Factory function to create a wrapped Mario environment."""
    def _init():
        env = MarioEnvWrapper(env_name)
        return env
    return _init


def train_ppo(env_name: str,
              total_timesteps: int,
              learning_rate: float,
              output_path: str):
    # Create vectorised environment
    env = make_vec_env(make_mario_env(env_name), n_envs=1)
    # Use MlpPolicy with default network architecture
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate, tensorboard_log="logs/ppo")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"Saved PPO model to {output_path}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO baseline on Mario environment.")
    parser.add_argument("--env", type=str, default="SuperMarioBros-1-1-v0", help="Environment name")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--output", type=str, required=True, help="Path to save PPO model")
    args = parser.parse_args()

    config = MarioConfig()
    timesteps = args.timesteps or config.ppo.total_timesteps
    lr = args.learning_rate or config.ppo.learning_rate

    train_ppo(args.env, timesteps, lr, args.output)


if __name__ == "__main__":
    main()