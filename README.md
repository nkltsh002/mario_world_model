# Mario World Model Project

This repository contains an implementation of the World Models architecture for the **Super Mario Bros** environment using the Gym Retro framework. The project is designed to let you build, train, and evaluate a world model on a classic video game, and to compare its performance to a model‑free baseline.

## Project Overview

World Models is a model‑based reinforcement learning approach introduced by Ha & Schmidhuber (2018).  It decomposes the agent into three components:

1. **Vision model (VAE)** – compresses high‑dimensional frames into a low‑dimensional latent vector.
2. **Memory model (MDN‑RNN)** – models the temporal dynamics in the latent space and predicts the next latent state conditioned on the previous latent state and action.
3. **Controller** – learns a policy in latent space using the hidden state from the memory model.

This repository focuses on training these components for the Super Mario Bros environment.  A Proximal Policy Optimisation (PPO) baseline is also provided for comparison.

## Repository Structure

```
mario_world_model/
├── config.py              # Hyper‑parameter configuration
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
├── models/
│   ├── vae.py             # Convolutional VAE implementation
│   ├── mdn_rnn.py         # Mixture Density Network RNN implementation
│   └── controller.py      # Simple controller network
├── utils/
│   ├── environment.py     # Mario environment wrapper and preprocessing
│   └── __init__.py
├── training/
│   ├── data_collection.py # Collect random rollouts from the environment
│   ├── train_vae.py       # Train the VAE on collected rollouts
│   ├── train_mdn_rnn.py   # Train the MDN‑RNN on latent sequences
│   ├── train_controller.py# Train the controller using CMA‑ES
│   ├── ppo_baseline.py    # Train a PPO agent for comparison
│   ├── evaluate.py        # Evaluate trained models
│   └── __init__.py
├── data/                  # Collected rollouts and processed datasets (empty)
├── checkpoints/           # Saved model weights (empty)
└── logs/                  # TensorBoard logs (empty)
```

Empty directories are created to organise your data and results.  You can add additional folders as needed.

## Getting Started

### 1. Install dependencies

Use a virtual environment (recommended) and install the requirements:

```bash
git clone <your-github-repository-url>
cd mario_world_model
pip install -r requirements.txt
# Install the Super Mario Bros environment
pip install gym-super-mario-bros nes-py
```

Gym Retro games require ROM files.  The `gym-super-mario-bros` package includes utilities to download and install the necessary ROM – follow the package instructions to set this up.

### 2. Collect rollouts

Before training, you need a dataset of observations and actions.  Run the data collection script to gather random rollouts:

```bash
python -m mario_world_model.training.data_collection --env SuperMarioBros-1-1-v0 \
    --episodes 50 \
    --output data/mario_rollouts.pkl
```

This will play the game with a random policy for 50 episodes and save a pickled dataset under `data/mario_rollouts.pkl`.  You can adjust the number of episodes and maximum steps per episode using command‑line flags.

### 3. Train the VAE

Train the variational autoencoder on the collected frames:

```bash
python -m mario_world_model.training.train_vae \
    --dataset data/mario_rollouts.pkl \
    --epochs 50 \
    --batch-size 64 \
    --output checkpoints/vae_mario.pt
```

The script loads the dataset, trains the VAE and saves the model weights to `checkpoints/vae_mario.pt`.  Use `--latent-size` and `--learning-rate` to override default hyper‑parameters.

### 4. Train the MDN‑RNN

Next, train the memory model on sequences of latent vectors:

```bash
python -m mario_world_model.training.train_mdn_rnn \
    --dataset data/mario_rollouts.pkl \
    --vae checkpoints/vae_mario.pt \
    --epochs 30 \
    --sequence-length 100 \
    --output checkpoints/mdn_rnn_mario.pt
```

This script encodes the collected frames using the VAE, forms sequences of latent vectors and actions, and trains an MDN‑RNN.  The resulting model weights are stored in `checkpoints/mdn_rnn_mario.pt`.

### 5. Train the controller

Finally, train the controller inside the latent world.  The controller uses CMA‑ES to search for a weight vector that maximises the predicted cumulative reward:

```bash
python -m mario_world_model.training.train_controller \
    --vae checkpoints/vae_mario.pt \
    --mdnrnn checkpoints/mdn_rnn_mario.pt \
    --iterations 200 \
    --population-size 64 \
    --output checkpoints/controller_mario.pt
```

At each iteration the script uses the MDN‑RNN to simulate trajectories in latent space and evaluates candidate controllers.  You can increase the number of iterations for better results at the cost of computation.

### 6. Train the PPO baseline (optional)

As a baseline, you can train a model‑free PPO agent directly on the environment:

```bash
python -m mario_world_model.training.ppo_baseline \
    --env SuperMarioBros-1-1-v0 \
    --timesteps 1000000 \
    --output checkpoints/ppo_mario
```

This will save the trained policy in the specified output directory.

### 7. Evaluate trained agents

After training, you can evaluate the world model and PPO agents:

```bash
python -m mario_world_model.training.evaluate \
    --vae checkpoints/vae_mario.pt \
    --mdnrnn checkpoints/mdn_rnn_mario.pt \
    --controller checkpoints/controller_mario.pt \
    --env SuperMarioBros-1-1-v0 \
    --episodes 10

python -m mario_world_model.training.evaluate \
    --ppo checkpoints/ppo_mario \
    --env SuperMarioBros-1-1-v0 \
    --episodes 10
```

These commands render the agent playing and report the mean score over the specified number of episodes.

## Notes

- Default hyper‑parameters are defined in `config.py`.  You can modify them directly or override them using command‑line flags.
- Training can be slow without a GPU.  The scripts use PyTorch and will automatically utilise CUDA if available.
- The controller training currently evaluates candidates in the latent world simulated by the MDN‑RNN; you can modify the script to evaluate in the real environment for better fidelity at increased cost.
- PPO training uses [stable‑baselines3](https://github.com/DLR-RM/stable-baselines3) and may require additional dependencies.

For background on the world model architecture refer to:

> D. Ha and J. Schmidhuber, “World Models,” *arXiv* preprint arXiv:1803.10122, 2018.

Feel free to customise and extend this repository for your honours project!