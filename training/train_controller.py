"""
Training script for the controller of the Mario World Model using CMA‑ES.

The controller is a small neural network that maps the current latent
state and memory model hidden state to an action.  It is optimised with
the CMA‑ES evolution strategy by directly maximising the cumulative
reward obtained in the environment.

Usage:
    python -m mario_world_model.training.train_controller \
        --vae checkpoints/vae_mario.pt \
        --mdnrnn checkpoints/mdn_rnn_mario.pt \
        --iterations 200 \
        --population-size 64 \
        --output checkpoints/controller_mario.pt
"""

import argparse
import os

import numpy as np
import torch

import cma

from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller
from utils.environment import MarioEnvWrapper
from config import MarioConfig


def get_param_vector(model: torch.nn.Module) -> torch.Tensor:
    """Flatten model parameters into a single vector."""
    return torch.nn.utils.parameters_to_vector(list(model.parameters())).detach()


def set_param_vector(model: torch.nn.Module, vector: torch.Tensor):
    """Assign a flat parameter vector to a model."""
    torch.nn.utils.vector_to_parameters(vector, model.parameters())


def evaluate_controller(controller: Controller,
                        vae: ConvVAE,
                        mdnrnn: MDNRNN,
                        env: MarioEnvWrapper,
                        episodes: int = 1,
                        max_steps: int = 1000) -> float:
    """
    Evaluate the controller by running it in the environment for a number
    of episodes and returning the average total reward.

    Args:
        controller: Controller network.
        vae: Trained VAE.
        mdnrnn: Trained MDN‑RNN (used only for hidden state).
        env: Wrapped Mario environment.
        episodes: Number of episodes to average over.
        max_steps: Maximum steps per episode.

    Returns:
        Mean total reward.
    """
    device = next(controller.parameters()).device
    vae.eval()
    mdnrnn.eval()
    total_reward = 0.0

    for _ in range(episodes):
        obs = env.reset()  # (frame_stack, H, W)
        # Convert to tensor and encode latent
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = vae.encode(obs_tensor)
        z = mu.squeeze(0)  # (latent_dim)
        # Initialise MDN-RNN hidden state (h0 and c0)
        h = None

        ep_reward = 0.0
        for t in range(max_steps):
            # If we have a hidden state from MDN-RNN, flatten it; else zeros
            if h is None:
                hidden_flat = torch.zeros((mdnrnn.hidden_size,), device=device)
            else:
                # h is a tuple (h_n, c_n), each (num_layers, batch, hidden_size)
                hidden_flat = h[0][-1].squeeze(0)
            # Concatenate latent vector and hidden state
            inp = torch.cat([z, hidden_flat], dim=-1).unsqueeze(0)  # (1, latent_dim + hidden_size)
            # Sample action
            with torch.no_grad():
                action = controller.act(inp)[0].item()
            # Step in environment
            next_obs, reward, done, _ = env.step(int(action))
            ep_reward += reward
            # Encode next observation
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, logvar = vae.encode(obs_tensor)
            next_z = mu.squeeze(0)
            # Update MDN-RNN hidden state using current latent and action
            # Form input: concatenate current latent and one-hot action
            a_one_hot = torch.zeros((1, mdnrnn.action_dim), device=device)
            a_one_hot[0, int(action)] = 1.0
            mdn_input = torch.cat([z.unsqueeze(0), a_one_hot], dim=-1)  # (1, latent_dim + action_dim)
            # Add sequence dimension for LSTM: (seq_len=1, batch=1, dim)
            mdn_input = mdn_input.unsqueeze(0)
            with torch.no_grad():
                _, h = mdnrnn.lstm(mdn_input, h)
            # Update latent
            z = next_z
            if done:
                break
        total_reward += ep_reward

    return total_reward / episodes


def train_controller(vae_path: str,
                     mdnrnn_path: str,
                     iterations: int,
                     population_size: int,
                     init_sigma: float,
                     output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MarioConfig()
    # Instantiate environment
    env = MarioEnvWrapper(config.env_name)
    # Load VAE
    frame_stack = env.frame_stack
    vae = ConvVAE(input_channels=frame_stack, latent_dim=config.vae.latent_size).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    # Load MDN-RNN
    mdnrnn = MDNRNN(latent_dim=config.vae.latent_size,
                    action_dim=env.action_space.n,
                    hidden_size=config.mdnrnn.hidden_size,
                    num_gaussians=config.mdnrnn.num_mixtures).to(device)
    mdnrnn.load_state_dict(torch.load(mdnrnn_path, map_location=device))
    # Build Controller
    input_dim = config.vae.latent_size + config.mdnrnn.hidden_size
    controller = Controller(input_dim=input_dim,
                            action_dim=env.action_space.n,
                            hidden_sizes=config.controller.hidden_sizes).to(device)

    # Initial parameter vector
    init_params = get_param_vector(controller).cpu().numpy()
    # CMA-ES strategy
    es = cma.CMAEvolutionStrategy(init_params, init_sigma, {'popsize': population_size})
    for iteration in range(1, iterations + 1):
        solutions = es.ask()
        rewards = []
        for sol in solutions:
            # Assign parameters
            set_param_vector(controller, torch.tensor(sol, dtype=torch.float32, device=device))
            # Evaluate
            reward = evaluate_controller(controller, vae, mdnrnn, env,
                                         episodes=1, max_steps=500)
            # CMA-ES minimises the objective, so use negative reward
            rewards.append(-reward)
        es.tell(solutions, rewards)
        best_reward = -min(rewards)
        print(f"Iteration {iteration}/{iterations}, best_reward={best_reward:.2f}")
    # Save the best found solution
    best_params = torch.tensor(es.result.xbest, dtype=torch.float32, device=device)
    set_param_vector(controller, best_params)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(controller.state_dict(), output_path)
    print(f"Saved controller to {output_path}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train controller for Mario World Model using CMA‑ES.")
    parser.add_argument("--vae", type=str, required=True, help="Path to trained VAE weights (.pt)")
    parser.add_argument("--mdnrnn", type=str, required=True, help="Path to trained MDN‑RNN weights (.pt)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of CMA‑ES iterations")
    parser.add_argument("--population-size", type=int, default=None, help="CMA‑ES population size")
    parser.add_argument("--sigma", type=float, default=None, help="Initial exploration noise sigma")
    parser.add_argument("--output", type=str, required=True, help="Path to save controller weights (.pt)")
    args = parser.parse_args()

    config = MarioConfig()
    iterations = args.iterations or config.controller.iterations
    popsize = args.population_size or config.controller.population_size
    sigma = args.sigma or config.controller.sigma
    train_controller(args.vae, args.mdnrnn, iterations, popsize, sigma, args.output)


if __name__ == "__main__":
    main()