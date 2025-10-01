"""
Evaluation script for trained world model components and PPO baseline.

This script evaluates either the world model (VAE + MDN‑RNN + controller) or
a trained PPO agent on the Super Mario Bros environment and reports the
average total reward over a number of episodes.  It can also render the
gameplay.

Usage:
    # Evaluate world model
    python -m mario_world_model.training.evaluate --vae checkpoints/vae_mario.pt \
        --mdnrnn checkpoints/mdn_rnn_mario.pt \
        --controller checkpoints/controller_mario.pt \
        --env SuperMarioBros-1-1-v0 --episodes 10

    # Evaluate PPO agent
    python -m mario_world_model.training.evaluate --ppo checkpoints/ppo_mario \
        --env SuperMarioBros-1-1-v0 --episodes 10
"""

import argparse

import torch

from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller
from utils.environment import MarioEnvWrapper
from config import MarioConfig

# Only import stable-baselines3 when needed
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def evaluate_world_model(vae_path: str,
                         mdnrnn_path: str,
                         controller_path: str,
                         env_name: str,
                         episodes: int = 10,
                         max_steps: int = 1000,
                         render: bool = False) -> float:
    """
    Evaluate a trained world model by running the controller in the real environment.

    Args:
        vae_path: Path to trained VAE weights (.pt).
        mdnrnn_path: Path to trained MDN‑RNN weights (.pt).
        controller_path: Path to trained controller weights (.pt).
        env_name: Environment to evaluate on.
        episodes: Number of episodes to average over.
        max_steps: Maximum steps per episode.
        render: Whether to render the environment (slow).

    Returns:
        Average total reward.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create environment
    env = MarioEnvWrapper(env_name)
    # Load models
    frame_stack = env.frame_stack
    config = MarioConfig()
    vae = ConvVAE(input_channels=frame_stack, latent_dim=config.vae.latent_size).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    mdnrnn = MDNRNN(latent_dim=config.vae.latent_size,
                    action_dim=env.action_space.n,
                    hidden_size=config.mdnrnn.hidden_size,
                    num_gaussians=config.mdnrnn.num_mixtures).to(device)
    mdnrnn.load_state_dict(torch.load(mdnrnn_path, map_location=device))
    controller = Controller(input_dim=config.vae.latent_size + config.mdnrnn.hidden_size,
                            action_dim=env.action_space.n,
                            hidden_sizes=config.controller.hidden_sizes).to(device)
    controller.load_state_dict(torch.load(controller_path, map_location=device))

    vae.eval()
    mdnrnn.eval()
    controller.eval()

    total_reward = 0.0
    for ep in range(episodes):
        obs = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = vae.encode(obs_tensor)
        z = mu.squeeze(0)
        h = None
        ep_reward = 0.0
        for t in range(max_steps):
            if h is None:
                hidden_flat = torch.zeros((mdnrnn.hidden_size,), device=device)
            else:
                hidden_flat = h[0][-1].squeeze(0)
            inp = torch.cat([z, hidden_flat], dim=-1).unsqueeze(0)
            with torch.no_grad():
                action = controller.act(inp, deterministic=True)[0].item()
            next_obs, reward, done, _ = env.step(int(action))
            ep_reward += reward
            if render:
                env.env.render()
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, _ = vae.encode(obs_tensor)
            next_z = mu.squeeze(0)
            # Update MDN-RNN hidden state
            a_one_hot = torch.zeros((1, mdnrnn.action_dim), device=device)
            a_one_hot[0, int(action)] = 1.0
            mdn_input = torch.cat([z.unsqueeze(0), a_one_hot], dim=-1).unsqueeze(0)
            with torch.no_grad():
                _, h = mdnrnn.lstm(mdn_input, h)
            z = next_z
            if done:
                break
        print(f"Episode {ep+1}/{episodes}: reward={ep_reward:.2f}")
        total_reward += ep_reward
    env.close()
    return total_reward / episodes


def evaluate_ppo(ppo_path: str,
                 env_name: str,
                 episodes: int = 10,
                 max_steps: int = 1000,
                 render: bool = False) -> float:
    """
    Evaluate a trained PPO agent on the environment.

    Args:
        ppo_path: Path to saved PPO agent.
        env_name: Environment name.
        episodes: Number of episodes to average over.
        max_steps: Maximum steps per episode.
        render: Whether to render the environment.

    Returns:
        Average total reward.
    """
    if PPO is None:
        raise ImportError("stable-baselines3 is required to evaluate PPO agents.")
    env = MarioEnvWrapper(env_name)
    # Load PPO
    model = PPO.load(ppo_path)
    total_reward = 0.0
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        for t in range(max_steps):
            # Flatten observation for MLP policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(int(action))
            ep_reward += reward
            if render:
                env.env.render()
            if done:
                break
        print(f"Episode {ep+1}/{episodes}: reward={ep_reward:.2f}")
        total_reward += ep_reward
    env.close()
    return total_reward / episodes


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Mario world model or PPO agent.")
    parser.add_argument("--vae", type=str, default=None, help="Path to trained VAE (.pt)")
    parser.add_argument("--mdnrnn", type=str, default=None, help="Path to trained MDN‑RNN (.pt)")
    parser.add_argument("--controller", type=str, default=None, help="Path to trained controller (.pt)")
    parser.add_argument("--ppo", type=str, default=None, help="Path to trained PPO agent")
    parser.add_argument("--env", type=str, default="SuperMarioBros-1-1-v0", help="Environment name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the gameplay")
    args = parser.parse_args()

    if args.ppo:
        mean_reward = evaluate_ppo(args.ppo, args.env, args.episodes, render=args.render)
        print(f"Average PPO reward over {args.episodes} episodes: {mean_reward:.2f}")
    else:
        if not (args.vae and args.mdnrnn and args.controller):
            raise ValueError("For world model evaluation you must provide --vae, --mdnrnn and --controller paths.")
        mean_reward = evaluate_world_model(args.vae, args.mdnrnn, args.controller,
                                           args.env, args.episodes, render=args.render)
        print(f"Average world model reward over {args.episodes} episodes: {mean_reward:.2f}")


if __name__ == "__main__":
    main()