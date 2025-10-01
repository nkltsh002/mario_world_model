"""
Data collection script for the Mario World Model project.

This script plays episodes of Super Mario Bros with a random policy and
records observations and actions.  The resulting dataset can be used to
train the VAE and MDN‑RNN.  Data is saved in pickle format as a list of
trajectories, where each trajectory is a dict with keys:
  - 'observations': list of stacked frames (np.ndarray, shape [frame_stack, H, W])
  - 'actions': list of integer actions
  - 'rewards': list of float rewards

Usage:
    python -m mario_world_model.training.data_collection --env SuperMarioBros-1-1-v0 \
        --episodes 50 --max-steps 1000 --output data/mario_rollouts.pkl
"""

import argparse
import os
import pickle
import random

import numpy as np

from utils.environment import MarioEnvWrapper


def collect_rollouts(env_name: str,
                     episodes: int,
                     max_steps: int,
                     output_path: str,
                     seed: int = 0):
    """
    Collect random rollouts from the specified Mario environment.

    Args:
        env_name: Name of the gym-super-mario-bros environment.
        episodes: Number of episodes to play.
        max_steps: Maximum steps per episode.
        output_path: Where to write the pickle file.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    env = MarioEnvWrapper(env_name)
    trajectories = []

    for ep in range(episodes):
        obs = env.reset()
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        total_reward = 0.0

        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            obs = next_obs
            total_reward += reward
            if done:
                break

        print(f"Episode {ep+1}/{episodes}: steps={len(episode_actions)}, total_reward={total_reward:.2f}")
        trajectories.append({
            "observations": np.stack(episode_obs, axis=0),
            "actions": np.array(episode_actions, dtype=np.int64),
            "rewards": np.array(episode_rewards, dtype=np.float32),
        })

    env.close()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect random rollouts for Mario world model training.")
    parser.add_argument("--env", type=str, default="SuperMarioBros-1-1-v0", help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--output", type=str, required=True, help="Path to save the dataset (pickle)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    collect_rollouts(args.env, args.episodes, args.max_steps, args.output, args.seed)


if __name__ == "__main__":
    main()