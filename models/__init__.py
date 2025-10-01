"""
Models package for the Mario World Model project.

This package contains the neural network modules used in the World Models
pipeline, including the variational autoencoder (VAE), the mixture density
network recurrent neural network (MDN‑RNN), and the controller.  Import
these modules from this package when constructing and training the
corresponding models.

Example:

    from models.vae import ConvVAE
    from models.mdn_rnn import MDNRNN
    from models.controller import Controller

"""

from .vae import ConvVAE  # noqa: F401
from .mdn_rnn import MDNRNN  # noqa: F401
from .controller import Controller  # noqa: F401