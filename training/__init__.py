"""
Training package for the Mario World Model project.

This package includes scripts and utilities to collect data, train the
VAE, MDN‑RNN and controller components of the World Models architecture,
train a PPO baseline, and evaluate trained agents.  You can run these
modules as scripts via the `python -m` interface.  For example:

    # Collect data using random actions
    python -m mario_world_model.training.data_collection --episodes 100 --output data/rollouts.pkl

    # Train the VAE on collected data
    python -m mario_world_model.training.train_vae --data data/rollouts.pkl --output checkpoints/vae.pt

    # Train the MDN‑RNN using latent vectors from the VAE
    python -m mario_world_model.training.train_mdn_rnn --data data/rollouts.pkl --vae checkpoints/vae.pt --output checkpoints/mdn_rnn.pt

    # Train the controller via CMA‑ES
    python -m mario_world_model.training.train_controller --vae checkpoints/vae.pt --mdn-rnn checkpoints/mdn_rnn.pt --output checkpoints/controller.pt

    # Train the PPO baseline
    python -m mario_world_model.training.ppo_baseline --env SuperMarioBros-1-1-v0 --timesteps 1000000 --output checkpoints/ppo_mario

See the README.md at the repository root for more details.
"""