"""Training subpackage for the Mario World Model project.

Each module can be executed via `python -m mario_world_model.training.<module>`.
Typical workflow:
    * Collect data: python -m mario_world_model.training.data_collection --episodes 100 --output data/rollouts.pkl
    * Train VAE: python -m mario_world_model.training.train_vae --dataset data/rollouts.pkl --output checkpoints/vae.pt
    * Train MDN-RNN: python -m mario_world_model.training.train_mdn_rnn --dataset data/rollouts.pkl --vae checkpoints/vae.pt --output checkpoints/mdn_rnn.pt
    * Optimise controller: python -m mario_world_model.training.train_controller --vae checkpoints/vae.pt --mdnrnn checkpoints/mdn_rnn.pt --output checkpoints/controller.pt
    * Optional PPO baseline: python -m mario_world_model.training.ppo_baseline --env SuperMarioBros-1-1-v0 --output checkpoints/ppo_mario
"""
